#!/usr/bin/env python3
"""
Inspect generated training data.

Loads a chunk specific HDF5 file and saves sample pairs as WAV files for inspection.

Usage:
    python scripts/inspect_generated_data.py --chunk data/generated/chunk_00000.h5 --output-dir preview
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    print("Error: h5py not installed. Run 'pip install h5py'")
    sys.exit(1)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Inspect generated training data")
    parser.add_argument("--chunk", type=Path, required=True,
                        help="Path to HDF5 chunk file")
    parser.add_argument("--output-dir", type=Path, default=Path("preview"),
                        help="Output directory for WAV files/plots")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random samples to inspect")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    if not args.chunk.exists():
        print(f"Error: Chunk file {args.chunk} not found")
        sys.exit(1)
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"INSPECTING: {args.chunk}")
    print("=" * 60)
    
    with h5py.File(args.chunk, 'r') as f:
        # Check attributes
        print("Attributes:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")
            
        mixtures = f['mixture']
        cleans = f['clean']
        
        n_total = len(mixtures)
        print(f"\nTotal samples: {n_total:,}")
        print(f"Mixture shape: {mixtures.shape}, dtype: {mixtures.dtype}")
        print(f"Clean shape: {cleans.shape}, dtype: {cleans.dtype}")
        
        # Select random indices
        np.random.seed(args.seed)
        indices = np.random.choice(n_total, args.num_samples, replace=False)
        indices.sort()
        
        print(f"\nExtracting {args.num_samples} samples: {indices}")
        
        for i, idx in enumerate(indices):
            # Load and convert to float32
            mix = mixtures[idx].astype(np.float32) / 32767.0
            clean = cleans[idx].astype(np.float32) / 32767.0
            
            # Compute stats
            mix_rms = np.sqrt(np.mean(mix**2))
            clean_rms = np.sqrt(np.mean(clean**2))
            si_sdr_val = 10 * np.log10(np.sum(clean**2) / (np.sum((mix - clean)**2) + 1e-8))
            
            print(f"\nSample {idx}:")
            print(f"  Mixture RMS: {mix_rms:.4f}")
            print(f"  Clean RMS: {clean_rms:.4f}")
            print(f"  SI-SDR (mix vs clean): {si_sdr_val:.2f} dB")
            
            # Save WAVs
            mix_path = args.output_dir / f"sample_{idx}_mix.wav"
            clean_path = args.output_dir / f"sample_{idx}_clean.wav"
            
            sf.write(mix_path, mix, 16000)
            sf.write(clean_path, clean, 16000)
            print(f"  Saved WAVs to {args.output_dir}")
            
            # Plot spectrograms if matplotlib available
            try:
                plt.figure(figsize=(10, 6))
                
                plt.subplot(2, 1, 1)
                plt.specgram(mix, NFFT=512, Fs=16000, noverlap=256)
                plt.title(f"Mixture (Sample {idx})")
                plt.ylabel("Frequency (Hz)")
                
                plt.subplot(2, 1, 2)
                plt.specgram(clean, NFFT=512, Fs=16000, noverlap=256)
                plt.title(f"Clean Target (Sample {idx})")
                plt.ylabel("Frequency (Hz)")
                plt.xlabel("Time (s)")
                
                plt.tight_layout()
                plt.savefig(args.output_dir / f"sample_{idx}_spec.png")
                plt.close()
                print(f"  Saved spectrogram to {args.output_dir / f'sample_{idx}_spec.png'}")
            except Exception as e:
                print(f"  Could not plot spectrogram: {e}")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
