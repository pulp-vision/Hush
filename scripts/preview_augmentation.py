#!/usr/bin/env python3
"""
Preview Augmentation Script.

Generates sample audio mixtures so you can listen and verify the augmentation
pipeline before training. Creates pairs of:
- mixture (noisy + background speaker)
- clean target (primary speaker only)
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf


def load_audio_robust(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file using soundfile (more robust than torchaudio for some formats)."""
    try:
        # Try soundfile first (handles more formats)
        audio, sr = sf.read(str(path), dtype='float32')
        
        # Convert to tensor
        if len(audio.shape) == 1:
            audio = audio[np.newaxis, :]  # [1, T]
        else:
            audio = audio.T  # [C, T]
            audio = audio.mean(axis=0, keepdims=True)  # Mono
        
        waveform = torch.from_numpy(audio)
        
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        
        return waveform
    except Exception as e:
        # Fallback to torchaudio
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform


def random_crop(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Randomly crop waveform to specified length."""
    if waveform.shape[-1] <= length:
        pad_size = length - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        start = random.randint(0, waveform.shape[-1] - length)
        waveform = waveform[..., start:start + length]
    return waveform


# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentation import MixingAugmentor


def main():
    parser = argparse.ArgumentParser(description="Preview augmentation samples")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ubuntu/Noise-Cancellation"),
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/samples"),
        help="Output directory for sample audio files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample mixtures to generate",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for audio",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=4.0,
        help="Length of audio segments in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Also save individual components (primary, background, noise)",
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifests
    manifest_dir = args.data_root / "data" / "manifests"
    
    print("Loading manifests...")
    with open(manifest_dir / "train.json") as f:
        train_manifest = json.load(f)
    with open(manifest_dir / "background.json") as f:
        background_manifest = json.load(f)
    with open(manifest_dir / "noise.json") as f:
        noise_manifest = json.load(f)
    
    primary_samples = train_manifest["samples"]
    background_samples = background_manifest["samples"]
    noise_samples = noise_manifest["samples"]
    
    print(f"Loaded {len(primary_samples)} primary, {len(background_samples)} background, {len(noise_samples)} noise samples")
    
    # Create augmentor with the same settings as training
    augmentor = MixingAugmentor(
        sample_rate=args.sample_rate,
        secondary_scale_range=(0.15, 0.5),
        pitch_shift_range=(-4.0, 4.0),
        noise_snr_range=(5.0, 20.0),
        p_secondary=1.0,  # Always add background for preview
        p_noise=1.0,       # Always add noise for preview
        p_pitch_shift=0.4,  # 40% get pitch shifted
    )
    
    segment_samples = int(args.segment_length * args.sample_rate)
    
    print(f"\nGenerating {args.num_samples} sample mixtures...")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}:")
        
        # Select random samples
        primary_info = random.choice(primary_samples)
        background_info = random.choice(background_samples)
        noise_info = random.choice(noise_samples)
        
        # Load audio
        primary_path = args.data_root / primary_info["path"]
        background_path = args.data_root / background_info["path"]
        noise_path = args.data_root / noise_info["path"]
        
        print(f"  Primary: {primary_info.get('speaker_id', 'unknown')} - {primary_path.name}")
        print(f"  Background: {background_info.get('speaker_id', 'unknown')} - {background_path.name}")
        print(f"  Noise: {noise_path.name}")
        
        try:
            primary = load_audio_robust(str(primary_path), args.sample_rate)
            background = load_audio_robust(str(background_path), args.sample_rate)
            noise = load_audio_robust(str(noise_path), args.sample_rate)
        except Exception as e:
            print(f"  Error loading audio: {e}")
            continue
        
        # Crop to segment length
        primary = random_crop(primary, segment_samples)
        background = random_crop(background, segment_samples)
        noise = random_crop(noise, segment_samples)
        
        # Create mixture using augmentor
        mixture, clean_target = augmentor(primary, background, noise)
        
        # Save audio files
        prefix = f"sample_{i+1:02d}"
        
        # Save mixture and clean target using soundfile
        sf.write(
            str(args.output_dir / f"{prefix}_mixture.wav"),
            mixture.squeeze(0).numpy(),
            args.sample_rate
        )
        sf.write(
            str(args.output_dir / f"{prefix}_clean.wav"),
            clean_target.squeeze(0).numpy(),
            args.sample_rate
        )
        
        print(f"  Saved: {prefix}_mixture.wav, {prefix}_clean.wav")
        
        # Optionally save individual components
        if args.save_components:
            sf.write(
                str(args.output_dir / f"{prefix}_primary.wav"),
                primary.squeeze(0).numpy(),
                args.sample_rate
            )
            sf.write(
                str(args.output_dir / f"{prefix}_background.wav"),
                background.squeeze(0).numpy(),
                args.sample_rate
            )
            sf.write(
                str(args.output_dir / f"{prefix}_noise.wav"),
                noise.squeeze(0).numpy(),
                args.sample_rate
            )
            print(f"  Saved components: {prefix}_primary.wav, {prefix}_background.wav, {prefix}_noise.wav")
    
    print("\n" + "=" * 60)
    print("PREVIEW GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {args.output_dir.absolute()}")
    print("\nListen to these pairs to verify augmentation:")
    print("  - *_mixture.wav: What the model will hear (primary + background + noise)")
    print("  - *_clean.wav: What the model should output (clean primary speaker)")
    print("\nThe model's job is to learn: mixture -> clean")
    print("=" * 60)


if __name__ == "__main__":
    main()
