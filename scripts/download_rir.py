#!/usr/bin/env python3
"""
Download SLR28 Room Impulse Response (RIR) dataset from OpenSLR.

This dataset contains simulated and real RIRs at 16kHz sample rate,
suitable for reverb augmentation in speech enhancement training.

Source: https://www.openslr.org/28/
Size: ~1.3GB

Usage:
    python3 scripts/download_rir.py --output-dir data/rir
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path


SLR28_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download file with progress indicator."""
    print(f"Downloading {url}")
    print(f"Destination: {dest}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\rProgress: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")
    
    print("\nDownload complete!")


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract zip file."""
    import zipfile
    print(f"Extracting {zip_path} to {dest_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    
    print("Extraction complete!")


def collect_rir_paths(rir_dir: Path) -> list:
    """Collect all RIR .wav files."""
    rir_files = list(rir_dir.rglob("*.wav"))
    # Filter to only include RIR files (not noise files)
    rir_files = [f for f in rir_files if "RIR" in str(f) or "rir" in str(f)]
    return rir_files


def main():
    parser = argparse.ArgumentParser(description="Download SLR28 RIR dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rir",
        help="Output directory for RIR files",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if already exists",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "rirs_noises.zip"
    
    # Download
    if not zip_path.exists() or not args.skip_download:
        download_file(SLR28_URL, zip_path)
    else:
        print(f"Using existing download: {zip_path}")
    
    # Extract
    extract_zip(zip_path, output_dir)
    
    # Collect RIR paths
    rir_files = collect_rir_paths(output_dir)
    print(f"\nFound {len(rir_files)} RIR files")
    
    # Save manifest
    manifest_path = output_dir / "rir_manifest.txt"
    with open(manifest_path, 'w') as f:
        for rir_path in sorted(rir_files):
            f.write(str(rir_path.absolute()) + "\n")
    
    print(f"Saved RIR manifest to: {manifest_path}")
    print(f"\nTo use in training, pass rir_paths to MixingAugmentor:")
    print(f"  rir_paths = open('{manifest_path}').read().strip().split('\\n')")


if __name__ == "__main__":
    main()
