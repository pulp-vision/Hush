"""
Fast in-memory dataset for pre-generated training data.

Loads a subset of data entirely into RAM for maximum training speed.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset


def load_pregenerated_data_to_memory(
    data_dir: Union[str, Path],
    max_samples: int = 100000,
    val_split: float = 0.05,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Load pre-generated data directly into memory for maximum speed.
    
    Args:
        data_dir: Directory with HDF5 chunks
        max_samples: Maximum samples to load
        val_split: Fraction for validation
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"Loading up to {max_samples:,} samples into memory...")
    
    all_mixtures = []
    all_cleans = []
    loaded = 0
    
    for chunk_name in manifest["chunks"]:
        if loaded >= max_samples:
            break
            
        chunk_path = data_dir / chunk_name
        if not chunk_path.exists():
            continue
            
        print(f"  Loading {chunk_name}...")
        with h5py.File(chunk_path, 'r') as f:
            n_in_chunk = len(f['mixture'])
            n_to_load = min(n_in_chunk, max_samples - loaded)
            
            # Load as float32 directly
            mix = f['mixture'][:n_to_load].astype(np.float32) / 32767.0
            clean = f['clean'][:n_to_load].astype(np.float32) / 32767.0
            
            all_mixtures.append(mix)
            all_cleans.append(clean)
            loaded += n_to_load
    
    # Concatenate all
    mixtures = np.concatenate(all_mixtures, axis=0)
    cleans = np.concatenate(all_cleans, axis=0)
    
    print(f"Loaded {len(mixtures):,} samples ({mixtures.nbytes / 1e9:.1f} GB)")
    
    # Shuffle
    indices = np.random.permutation(len(mixtures))
    mixtures = mixtures[indices]
    cleans = cleans[indices]
    
    # Split
    val_size = int(len(mixtures) * val_split)
    train_size = len(mixtures) - val_size
    
    # Convert to tensors with channel dimension
    train_mix = torch.from_numpy(mixtures[:train_size]).unsqueeze(1)  # [N, 1, T]
    train_clean = torch.from_numpy(cleans[:train_size]).unsqueeze(1)
    val_mix = torch.from_numpy(mixtures[train_size:]).unsqueeze(1)
    val_clean = torch.from_numpy(cleans[train_size:]).unsqueeze(1)
    
    train_dataset = TensorDataset(train_mix, train_clean)
    val_dataset = TensorDataset(val_mix, val_clean)
    
    print(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")
    
    return train_dataset, val_dataset


def create_fast_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 256,
    max_samples: int = 100000,
    num_workers: int = 0,
    val_split: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create fast dataloaders with in-memory data.
    
    Args:
        data_dir: Directory with HDF5 chunks
        batch_size: Batch size
        max_samples: Maximum samples to load into memory
        num_workers: Number of workers (0 for in-memory data)
        val_split: Validation split fraction
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset, val_dataset = load_pregenerated_data_to_memory(
        data_dir, max_samples, val_split
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
