"""
Dataset classes for DeepFilterNet-SE training.

Provides PyTorch datasets that load audio files and create synthetic mixtures
on-the-fly for training joint denoising + speaker extraction.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .augmentation import (
    MixingAugmentor,
    load_audio,
    random_crop,
)


class SpeakerMixtureDataset(Dataset):
    """
    Dataset for training DeepFilterNet-SE with synthetic speaker mixtures.
    
    Each sample consists of:
    - mixture: primary speaker + background speaker (pitch-shifted) + noise
    - target: clean primary speaker
    
    The model learns to output the clean primary speaker given the mixture,
    effectively learning both denoising and background speaker suppression.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        sample_rate: int = 16000,
        segment_length: float = 3.0,  # seconds
        augmentor: Optional[MixingAugmentor] = None,
        noise_manifest_path: Optional[Union[str, Path]] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            manifest_path: Path to JSON manifest with speaker audio files
            sample_rate: Target sample rate
            segment_length: Length of audio segments in seconds
            augmentor: MixingAugmentor instance for creating mixtures
            noise_manifest_path: Optional path to noise file manifest
            return_metadata: If True, return additional metadata dict
        """
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.augmentor = augmentor or MixingAugmentor(sample_rate=sample_rate)
        self.return_metadata = return_metadata
        
        # Load manifests
        self.samples = self._load_manifest(self.manifest_path)
        
        # Group samples by speaker for selecting different speakers
        self.speaker_to_samples: Dict[str, List[Dict]] = {}
        for sample in self.samples:
            spk = sample.get("speaker_id", "unknown")
            if spk not in self.speaker_to_samples:
                self.speaker_to_samples[spk] = []
            self.speaker_to_samples[spk].append(sample)
        
        self.speaker_ids = list(self.speaker_to_samples.keys())
        
        # Load noise manifest if provided
        self.noise_samples = []
        if noise_manifest_path is not None:
            self.noise_samples = self._load_manifest(Path(noise_manifest_path))
    
    def _load_manifest(self, path: Path) -> List[Dict]:
        """Load JSON manifest file."""
        with open(path, "r") as f:
            manifest = json.load(f)
        # Handle both list format and dict with "samples" key
        if isinstance(manifest, dict):
            return manifest.get("samples", [])
        return manifest
    
    def _get_different_speaker_sample(self, exclude_speaker: str) -> Optional[Dict]:
        """Get a random sample from a different speaker."""
        available_speakers = [s for s in self.speaker_ids if s != exclude_speaker]
        if not available_speakers:
            return None
        other_speaker = random.choice(available_speakers)
        return random.choice(self.speaker_to_samples[other_speaker])
    
    def _get_noise_sample(self) -> Optional[Tensor]:
        """Get a random noise sample."""
        if not self.noise_samples:
            return None
        noise_info = random.choice(self.noise_samples)
        try:
            noise = load_audio(noise_info["path"], self.sample_rate)
            return noise
        except Exception:
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Dict]]:
        """
        Get a training sample.
        
        Returns:
            If return_metadata is False:
                Tuple of (mixture, clean_primary), each [1, T]
            If return_metadata is True:
                Tuple of (mixture, clean_primary, metadata_dict)
        """
        sample = self.samples[idx]
        
        # Load primary speaker
        primary_path = sample["path"]
        primary_speaker = sample.get("speaker_id", "unknown")
        
        try:
            primary = load_audio(primary_path, self.sample_rate)
        except Exception as e:
            # Fallback to a random other sample if loading fails
            print(f"Warning: Failed to load {primary_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Crop to segment length
        primary = random_crop(primary, self.segment_samples)
        
        # Get background speaker from different speaker
        secondary = None
        secondary_info = self._get_different_speaker_sample(primary_speaker)
        if secondary_info is not None:
            try:
                secondary = load_audio(secondary_info["path"], self.sample_rate)
                secondary = random_crop(secondary, self.segment_samples)
            except Exception:
                secondary = None
        
        # Get noise
        noise = self._get_noise_sample()
        if noise is not None:
            noise = random_crop(noise, self.segment_samples)
        
        # Create mixture using augmentor
        mixture, clean_primary = self.augmentor(
            primary=primary,
            secondary=secondary,
            noise=noise,
        )
        
        if self.return_metadata:
            metadata = {
                "primary_path": primary_path,
                "primary_speaker": primary_speaker,
                "has_secondary": secondary is not None,
                "has_noise": noise is not None,
            }
            return mixture, clean_primary, metadata
        
        return mixture, clean_primary


class DebugDataset(Dataset):
    """
    Small debug dataset for fast iteration.
    
    Can be created from a full dataset by taking a subset.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        max_samples: int = 1000,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.max_samples = min(max_samples, len(base_dataset))
        
        # Create reproducible subset indices
        rng = random.Random(seed)
        self.indices = rng.sample(range(len(base_dataset)), self.max_samples)
    
    def __len__(self) -> int:
        return self.max_samples
    
    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """
    Collate function for DataLoader.
    
    Stacks batch of (mixture, target) tuples into batched tensors.
    """
    mixtures, targets = zip(*batch)
    return torch.stack(mixtures), torch.stack(targets)


def create_dataloaders(
    train_manifest: Union[str, Path],
    val_manifest: Union[str, Path],
    noise_manifest: Optional[Union[str, Path]] = None,
    rir_manifest: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    sample_rate: int = 16000,
    segment_length: float = 3.0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_manifest: Path to training manifest JSON
        val_manifest: Path to validation manifest JSON
        noise_manifest: Optional path to noise manifest JSON
        rir_manifest: Optional path to RIR manifest (txt file with paths)
        batch_size: Batch size
        num_workers: Number of data loading workers
        sample_rate: Target sample rate
        segment_length: Audio segment length in seconds
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load RIR paths if provided
    rir_paths = []
    if rir_manifest is not None:
        rir_manifest = Path(rir_manifest)
        if rir_manifest.exists():
            with open(rir_manifest, 'r') as f:
                rir_paths = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(rir_paths)} RIR paths for reverb augmentation")
    
    train_augmentor = MixingAugmentor(
        sample_rate=sample_rate,
        p_secondary=0.8,
        p_noise=0.9,
        p_reverb=0.2 if rir_paths else 0.0,  # Match DFN3: p_reverb=0.2
        rir_paths=rir_paths,
    )
    
    val_augmentor = MixingAugmentor(
        sample_rate=sample_rate,
        secondary_scale_range=(0.3, 0.3),  # Fixed for reproducibility
        pitch_shift_range=(0.0, 0.0),  # No pitch shift for validation
        noise_snr_range=(10.0, 10.0),
        p_secondary=1.0,  # Always add for validation
        p_noise=1.0,
        p_pitch_shift=0.0,  # No pitch shift during validation
        p_reverb=0.0,  # No reverb for validation (deterministic)
        rir_paths=[],
    )
    
    train_dataset = SpeakerMixtureDataset(
        manifest_path=train_manifest,
        sample_rate=sample_rate,
        segment_length=segment_length,
        augmentor=train_augmentor,
        noise_manifest_path=noise_manifest,
    )
    
    val_dataset = SpeakerMixtureDataset(
        manifest_path=val_manifest,
        sample_rate=sample_rate,
        segment_length=segment_length,
        augmentor=val_augmentor,
        noise_manifest_path=noise_manifest,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader

