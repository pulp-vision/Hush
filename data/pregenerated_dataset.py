"""
Dataset class for pre-generated training data.

Loads (mixture, clean) pairs from HDF5 files for fast training.
Uses direct HDF5 indexing for memory-efficient access.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class PreGeneratedDataset(Dataset):
    """
    Dataset for pre-generated (mixture, clean) pairs stored in HDF5 files.
    
    Uses direct HDF5 indexing for memory-efficient access.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        max_samples: Optional[int] = None,
        shuffle_chunks: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing chunk_*.h5 files and manifest.json
            max_samples: Optional limit on number of samples
            shuffle_chunks: Whether to shuffle chunk order
        """
        self.data_dir = Path(data_dir)
        
        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        self.sample_rate = self.manifest["sample_rate"]
        self.segment_length = self.manifest["segment_length"]
        self.chunk_size = self.manifest["chunk_size"]
        
        # Build index: list of (chunk_path, local_idx) for each global sample
        self.sample_index = []
        
        for chunk_name in self.manifest["chunks"]:
            chunk_path = self.data_dir / chunk_name
            if chunk_path.exists():
                with h5py.File(chunk_path, 'r') as f:
                    n_samples = f.attrs.get('num_samples', len(f['mixture']))
                for local_idx in range(n_samples):
                    self.sample_index.append((str(chunk_path), local_idx))
        
        self.total_samples = len(self.sample_index)
        if max_samples is not None:
            self.total_samples = min(self.total_samples, max_samples)
            self.sample_index = self.sample_index[:self.total_samples]
        
        print(f"Loaded {len(self.manifest['chunks'])} chunks, {self.total_samples:,} samples")
        
        # Shuffle samples for variety
        if shuffle_chunks:
            random.shuffle(self.sample_index)
        
        # Cache for open HDF5 file handles (one at a time)
        self._current_chunk_path = None
        self._current_file = None
    
    def _get_file(self, chunk_path: str):
        """Get HDF5 file handle, caching for efficiency."""
        if self._current_chunk_path != chunk_path:
            if self._current_file is not None:
                self._current_file.close()
            self._current_file = h5py.File(chunk_path, 'r')
            self._current_chunk_path = chunk_path
        return self._current_file
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get a training sample using direct HDF5 indexing (no full array load).
        
        Returns:
            Tuple of (mixture, clean), each [1, T] tensor
        """
        chunk_path, local_idx = self.sample_index[idx]
        f = self._get_file(chunk_path)
        
        # Direct indexing - only reads the specific sample, not entire array
        mixture = f['mixture'][local_idx].astype(np.float32) / 32767.0
        clean = f['clean'][local_idx].astype(np.float32) / 32767.0
        
        # Convert to tensors with channel dim
        mixture = torch.from_numpy(mixture).unsqueeze(0)  # [1, T]
        clean = torch.from_numpy(clean).unsqueeze(0)  # [1, T]
        
        return mixture, clean
    
    def __del__(self):
        """Close any open file handles."""
        if self._current_file is not None:
            try:
                self._current_file.close()
            except:
                pass


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """Collate function for DataLoader."""
    mixtures, cleans = zip(*batch)
    return torch.stack(mixtures), torch.stack(cleans)


def create_pregenerated_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 0,  # Default to 0 for HDF5 (not fork-safe)
    val_split: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for pre-generated data.
    
    Args:
        train_dir: Directory with training data
        val_dir: Optional separate validation directory
        batch_size: Batch size
        num_workers: Number of data loading workers (0 recommended for HDF5)
        val_split: Fraction of training data to use for validation if no val_dir
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = PreGeneratedDataset(train_dir)
    
    if val_dir is not None:
        val_dataset = PreGeneratedDataset(val_dir)
    else:
        # Split training data
        total = len(train_dataset)
        val_size = int(total * val_split)
        train_size = total - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader
