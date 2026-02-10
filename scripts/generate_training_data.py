#!/usr/bin/env python3
"""
Pre-generate training data for DeepFilterNet-SE.

Creates mixture + clean pairs and saves them to HDF5 files for fast training.
Uses multiprocessing for parallel generation.

Usage:
    python scripts/generate_training_data.py --output-dir data/generated --num-samples 500000 --workers 8
"""

import argparse
import json
import os
import random
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np

# We'll import audio functions inline to avoid multiprocessing issues


def load_audio_sf(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio using soundfile."""
    import soundfile as sf
    import torchaudio
    
    try:
        audio, sr = sf.read(str(path), dtype='float32')
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Mono
        
        # Resample if needed
        if sr != target_sr:
            import torch
            waveform = torch.from_numpy(audio).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            audio = waveform.squeeze(0).numpy()
        
        return audio.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


def random_crop(audio: np.ndarray, length: int) -> np.ndarray:
    """Randomly crop or pad audio to target length."""
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start:start + length]
    else:
        # Pad with zeros
        padded = np.zeros(length, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded


def compute_rms(audio: np.ndarray, eps: float = 1e-8) -> float:
    """Compute RMS energy."""
    return float(np.sqrt(np.mean(audio ** 2) + eps))


def mix_audio(
    primary: np.ndarray,
    secondary: Optional[np.ndarray],
    noise: Optional[np.ndarray],
    secondary_scale_range: Tuple[float, float] = (0.15, 0.5),
    noise_snr_range: Tuple[float, float] = (5.0, 20.0),
    p_secondary: float = 0.8,
    p_noise: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create mixture from primary + secondary + noise."""
    
    primary_rms = compute_rms(primary)
    mixture = primary.copy()
    
    # Add secondary speaker
    if secondary is not None and random.random() < p_secondary:
        scale = random.uniform(*secondary_scale_range)
        secondary_scaled = secondary * (primary_rms * scale / (compute_rms(secondary) + 1e-8))
        mixture = mixture + secondary_scaled
    
    # Add noise
    if noise is not None and random.random() < p_noise:
        target_snr = random.uniform(*noise_snr_range)
        noise_rms = compute_rms(noise)
        target_noise_rms = primary_rms / (10 ** (target_snr / 20))
        noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-8))
        mixture = mixture + noise_scaled
    
    # Normalize to prevent clipping
    max_val = max(np.abs(mixture).max(), np.abs(primary).max(), 1e-8)
    if max_val > 0.99:
        scale = 0.95 / max_val
        mixture = mixture * scale
        primary = primary * scale
    
    return mixture.astype(np.float32), primary.astype(np.float32)


class DataGenerator:
    """Generates training samples from manifests."""
    
    def __init__(
        self,
        primary_manifest: Path,
        background_manifest: Path,
        noise_manifest: Path,
        data_root: Path,
        sample_rate: int = 16000,
        segment_length: float = 3.0,
    ):
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        
        # Load manifests
        print("Loading manifests...")
        self.primary_samples = self._load_manifest(primary_manifest)
        self.background_samples = self._load_manifest(background_manifest)
        self.noise_samples = self._load_manifest(noise_manifest)
        
        print(f"  Primary speakers: {len(self.primary_samples)}")
        print(f"  Background speakers: {len(self.background_samples)}")
        print(f"  Noise samples: {len(self.noise_samples)}")
    
    def _load_manifest(self, path: Path) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("samples", [])
        return data
    
    def generate_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single (mixture, clean) pair."""
        # Select random samples
        primary_info = random.choice(self.primary_samples)
        background_info = random.choice(self.background_samples)
        noise_info = random.choice(self.noise_samples)
        
        # Load audio
        primary_path = self.data_root / primary_info["path"]
        background_path = self.data_root / background_info["path"]
        noise_path = self.data_root / noise_info["path"]
        
        primary = load_audio_sf(str(primary_path), self.sample_rate)
        background = load_audio_sf(str(background_path), self.sample_rate)
        noise = load_audio_sf(str(noise_path), self.sample_rate)
        
        # Crop to segment length
        primary = random_crop(primary, self.segment_samples)
        background = random_crop(background, self.segment_samples)
        noise = random_crop(noise, self.segment_samples)
        
        # Create mixture
        mixture, clean = mix_audio(primary, background, noise)
        
        return mixture, clean


# Global generator for multiprocessing
_generator = None


def init_worker(primary_manifest, background_manifest, noise_manifest, data_root, sample_rate, segment_length):
    """Initialize worker with generator."""
    global _generator
    _generator = DataGenerator(
        Path(primary_manifest),
        Path(background_manifest),
        Path(noise_manifest),
        Path(data_root),
        sample_rate,
        segment_length,
    )


def generate_one(idx: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Generate one sample (for multiprocessing)."""
    try:
        mixture, clean = _generator.generate_sample(idx)
        return idx, mixture, clean
    except Exception as e:
        print(f"Error generating sample {idx}: {e}")
        return idx, None, None


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"),
                        help="Output directory for generated data")
    parser.add_argument("--num-samples", type=int, default=500000,
                        help="Number of samples to generate")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Samples per HDF5 file")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-length", type=float, default=3.0)
    parser.add_argument("--data-root", type=Path, 
                        default=Path("/home/ubuntu/Noise-Cancellation"))
    parser.add_argument("--manifest-dir", type=Path,
                        default=Path("/home/ubuntu/Noise-Cancellation/data/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.workers is None:
        args.workers = cpu_count()
    
    print("=" * 60)
    print("TRAINING DATA GENERATION")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Samples to generate: {args.num_samples:,}")
    print(f"Workers: {args.workers}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Segment length: {args.segment_length}s")
    print(f"Sample rate: {args.sample_rate} Hz")
    
    # Calculate expected storage
    samples_per_file = args.chunk_size
    bytes_per_sample = args.segment_length * args.sample_rate * 2 * 2  # mixture + clean, int16
    total_bytes = args.num_samples * bytes_per_sample
    print(f"Estimated storage: {total_bytes / 1e9:.1f} GB")
    print("=" * 60)
    
    # Manifest paths
    primary_manifest = args.manifest_dir / "train.json"
    background_manifest = args.manifest_dir / "background.json"
    noise_manifest = args.manifest_dir / "noise.json"
    
    # Create chunks
    num_chunks = (args.num_samples + args.chunk_size - 1) // args.chunk_size
    print(f"\nGenerating {num_chunks} chunks...")
    
    start_time = time.time()
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, args.num_samples)
        chunk_samples = chunk_end - chunk_start
        
        chunk_file = args.output_dir / f"chunk_{chunk_idx:05d}.h5"
        
        if chunk_file.exists():
            print(f"Chunk {chunk_idx + 1}/{num_chunks} already exists, skipping...")
            continue
        
        print(f"\nChunk {chunk_idx + 1}/{num_chunks}: samples {chunk_start:,} to {chunk_end:,}")
        
        # Initialize worker pool
        with Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(
                str(primary_manifest),
                str(background_manifest),
                str(noise_manifest),
                str(args.data_root),
                args.sample_rate,
                args.segment_length,
            ),
        ) as pool:
            # Generate samples in parallel
            results = []
            for result in pool.imap_unordered(generate_one, range(chunk_samples), chunksize=100):
                results.append(result)
                if len(results) % 1000 == 0:
                    elapsed = time.time() - start_time
                    total_done = chunk_start + len(results)
                    rate = total_done / elapsed
                    eta = (args.num_samples - total_done) / rate if rate > 0 else 0
                    print(f"  Progress: {total_done:,}/{args.num_samples:,} "
                          f"({100*total_done/args.num_samples:.1f}%) "
                          f"Rate: {rate:.1f}/s, ETA: {eta/3600:.1f}h")
        
        # Filter successful results and sort by index
        valid_results = [(i, m, c) for i, m, c in results if m is not None]
        valid_results.sort(key=lambda x: x[0])
        
        if not valid_results:
            print(f"  WARNING: No valid samples in chunk {chunk_idx}")
            continue
        
        # Save to HDF5
        segment_len = int(args.segment_length * args.sample_rate)
        
        with h5py.File(chunk_file, 'w') as f:
            mixtures = np.stack([m for _, m, _ in valid_results])
            cleans = np.stack([c for _, _, c in valid_results])
            
            # Convert to int16 for storage efficiency
            mixtures_int16 = (mixtures * 32767).astype(np.int16)
            cleans_int16 = (cleans * 32767).astype(np.int16)
            
            f.create_dataset('mixture', data=mixtures_int16, compression='gzip', compression_opts=1)
            f.create_dataset('clean', data=cleans_int16, compression='gzip', compression_opts=1)
            f.attrs['sample_rate'] = args.sample_rate
            f.attrs['segment_length'] = args.segment_length
            f.attrs['num_samples'] = len(valid_results)
        
        print(f"  Saved {len(valid_results)} samples to {chunk_file}")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Output directory: {args.output_dir}")
    print(f"Total chunks: {num_chunks}")
    
    # Create manifest for generated data
    manifest = {
        "num_samples": args.num_samples,
        "chunk_size": args.chunk_size,
        "num_chunks": num_chunks,
        "sample_rate": args.sample_rate,
        "segment_length": args.segment_length,
        "chunks": [f"chunk_{i:05d}.h5" for i in range(num_chunks)],
    }
    
    with open(args.output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
