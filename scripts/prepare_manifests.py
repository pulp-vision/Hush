#!/usr/bin/env python3
"""
Prepare data manifests for DeepFilterNet-SE training.

Data Partitioning Strategy:
- PRIMARY SPEAKERS: VCTK + LibriSpeech (English, speaker-labeled)
- BACKGROUND SPEAKERS: DNS4 non-English speech (French, German, Italian, Spanish, Russian)
- NOISE: DNS4 noise_fullband + MUSAN
- IMPULSE RESPONSES: DNS4 impulse_responses

This ensures zero speaker overlap between primary and background pools.

Usage:
    python scripts/prepare_manifests.py --data-root /path/to/datasets --output-dir data/manifests
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import soundfile as sf
from tqdm import tqdm

# Non-English subdirectories in DNS4 to use as background speakers
BACKGROUND_SPEAKER_DIRS = [
    "french_speech",
    "german_speech",
    "italian_speech",
    "spanish_speech",
    "russian_speech",
]


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds."""
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 0.0


def index_vctk(data_root: Path) -> Tuple[List[Dict], Set[str]]:
    """
    Index VCTK dataset.
    
    Expected structure:
    data_root/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/*.flac
    """
    samples = []
    speakers = set()
    
    vctk_path = data_root / "VCTK-Corpus-0.92" / "wav48_silence_trimmed"
    if not vctk_path.exists():
        # Try alternative path
        vctk_path = data_root / "VCTK" / "wav48_silence_trimmed"
    
    if not vctk_path.exists():
        # Try direct path
        vctk_path = data_root / "wav48_silence_trimmed"

    if not vctk_path.exists():
        print(f"Warning: VCTK not found at {vctk_path}")
        return samples, speakers
    
    for speaker_dir in tqdm(list(vctk_path.iterdir()), desc="Indexing VCTK"):
        if not speaker_dir.is_dir():
            continue
        speaker_id = f"vctk_{speaker_dir.name}"
        speakers.add(speaker_id)
        
        for audio_file in speaker_dir.glob("*.flac"):
            duration = get_audio_duration(str(audio_file))
            if duration < 1.0:  # Skip very short files
                continue
            samples.append({
                "path": str(audio_file),
                "speaker_id": speaker_id,
                "duration": duration,
                "dataset": "vctk",
            })
    
    print(f"VCTK: {len(samples)} samples, {len(speakers)} speakers")
    return samples, speakers


def index_dns4_background(data_root: Path) -> Tuple[List[Dict], Set[str]]:
    """
    Index DNS4 non-English speech as background speakers.
    
    Expected structure:
    data_root/dns4/datasets_fullband/clean_fullband/{french_speech,german_speech,...}/*.wav
    """
    samples = []
    speakers = set()
    
    clean_path = data_root / "dns4" / "datasets_fullband" / "clean_fullband"
    if not clean_path.exists():
        print(f"Warning: DNS4 clean_fullband not found at {clean_path}")
        return samples, speakers
    
    for lang_dir in BACKGROUND_SPEAKER_DIRS:
        lang_path = clean_path / lang_dir
        if not lang_path.exists():
            print(f"Warning: {lang_dir} not found at {lang_path}")
            continue
        
        # Use glob for efficiency with large directories
        audio_files = list(lang_path.rglob("*.wav"))
        print(f"Found {len(audio_files)} files in {lang_dir}")
        
        for audio_file in tqdm(audio_files, desc=f"Indexing {lang_dir}"):
            duration = get_audio_duration(str(audio_file))
            if duration < 1.0:
                continue
            
            # Use parent directory + file stem as speaker ID
            parent_name = audio_file.parent.name
            speaker_id = f"bg_{lang_dir}_{parent_name}"
            speakers.add(speaker_id)
            
            samples.append({
                "path": str(audio_file),
                "speaker_id": speaker_id,
                "duration": duration,
                "dataset": "dns4_background",
                "language": lang_dir.replace("_speech", ""),
            })
    
    print(f"DNS4 background: {len(samples)} samples, {len(speakers)} speakers")
    return samples, speakers


def index_dns4_noise(data_root: Path) -> List[Dict]:
    """
    Index DNS4 noise files.
    
    Expected structure:
    data_root/dns4/datasets_fullband/noise_fullband/*.wav
    """
    samples = []
    
    noise_path = data_root / "dns4" / "datasets_fullband" / "noise_fullband"
    if not noise_path.exists():
        print(f"Warning: DNS4 noise not found at {noise_path}")
        return samples
    
    audio_files = list(noise_path.rglob("*.wav"))
    print(f"Found {len(audio_files)} noise files")
    
    for audio_file in tqdm(audio_files, desc="Indexing DNS4 noise"):
        duration = get_audio_duration(str(audio_file))
        if duration < 1.0:
            continue
        samples.append({
            "path": str(audio_file),
            "duration": duration,
            "category": audio_file.parent.name,
        })
    
    print(f"DNS4 noise: {len(samples)} samples")
    return samples


def index_musan(data_root: Path) -> List[Dict]:
    """
    Index MUSAN noise files.
    
    Expected structure:
    data_root/musan/{noise,music,speech}/**/*.wav
    """
    samples = []
    
    musan_path = data_root / "musan"
    if not musan_path.exists():
        print(f"Warning: MUSAN not found at {musan_path}")
        return samples
    
    # Only use noise and music categories (not speech)
    for category in ["noise", "music"]:
        cat_path = musan_path / category
        if not cat_path.exists():
            continue
        
        audio_files = list(cat_path.rglob("*.wav"))
        for audio_file in tqdm(audio_files, desc=f"Indexing MUSAN {category}"):
            duration = get_audio_duration(str(audio_file))
            if duration < 1.0:
                continue
            samples.append({
                "path": str(audio_file),
                "duration": duration,
                "category": category,
                "dataset": "musan",
            })
    
    print(f"MUSAN noise: {len(samples)} samples")
    return samples


def index_impulse_responses(data_root: Path) -> List[Dict]:
    """
    Index impulse response files.
    
    Expected structure:
    data_root/dns4/datasets_fullband/impulse_responses/**/*.wav
    """
    samples = []
    
    ir_path = data_root / "dns4" / "datasets_fullband" / "impulse_responses"
    if not ir_path.exists():
        print(f"Warning: Impulse responses not found at {ir_path}")
        return samples
    
    audio_files = list(ir_path.rglob("*.wav"))
    for audio_file in tqdm(audio_files, desc="Indexing impulse responses"):
        duration = get_audio_duration(str(audio_file))
        if duration < 0.01:  # IRs can be very short
            continue
        samples.append({
            "path": str(audio_file),
            "duration": duration,
        })
    
    print(f"Impulse responses: {len(samples)} samples")
    return samples


def index_librispeech(data_root: Path) -> Tuple[List[Dict], Set[str]]:
    """
    Index LibriSpeech dataset (as alternative to VCTK).
    
    Expected structure:
    data_root/LibriSpeech/train-clean-100/speaker_id/chapter_id/*.flac
    """
    samples = []
    speakers = set()
    
    for subset in ["train-clean-100", "train-clean-360", "dev-clean", "test-clean"]:
        subset_path = data_root / "LibriSpeech" / subset
        if not subset_path.exists():
            continue
        
        for speaker_dir in tqdm(list(subset_path.iterdir()), desc=f"Indexing LibriSpeech {subset}"):
            if not speaker_dir.is_dir():
                continue
            speaker_id = f"libri_{speaker_dir.name}"
            speakers.add(speaker_id)
            
            for audio_file in speaker_dir.rglob("*.flac"):
                duration = get_audio_duration(str(audio_file))
                if duration < 1.0:
                    continue
                samples.append({
                    "path": str(audio_file),
                    "speaker_id": speaker_id,
                    "duration": duration,
                    "dataset": "librispeech",
                })
    
    print(f"LibriSpeech: {len(samples)} samples, {len(speakers)} speakers")
    return samples, speakers


def create_splits(
    samples: List[Dict],
    speakers: Set[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create speaker-disjoint train/val/test splits.
    
    Speakers are split, not individual samples, ensuring no speaker
    appears in multiple splits.
    """
    random.seed(seed)
    
    speaker_list = list(speakers)
    random.shuffle(speaker_list)
    
    n_speakers = len(speaker_list)
    n_train = int(n_speakers * train_ratio)
    n_val = int(n_speakers * val_ratio)
    
    train_speakers = set(speaker_list[:n_train])
    val_speakers = set(speaker_list[n_train:n_train + n_val])
    test_speakers = set(speaker_list[n_train + n_val:])
    
    train_samples = [s for s in samples if s["speaker_id"] in train_speakers]
    val_samples = [s for s in samples if s["speaker_id"] in val_speakers]
    test_samples = [s for s in samples if s["speaker_id"] in test_speakers]
    
    print(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    print(f"Speakers: {len(train_speakers)} train, {len(val_speakers)} val, {len(test_speakers)} test")
    
    return train_samples, val_samples, test_samples


def create_debug_subset(
    samples: List[Dict],
    max_hours: float = 3.0,
    seed: int = 42,
) -> List[Dict]:
    """Create a small debug subset for fast iteration."""
    random.seed(seed)
    
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    subset = []
    total_duration = 0.0
    max_seconds = max_hours * 3600
    
    for sample in shuffled:
        if total_duration >= max_seconds:
            break
        subset.append(sample)
        total_duration += sample.get("duration", 10.0)
    
    print(f"Debug subset: {len(subset)} samples, {total_duration/3600:.1f} hours")
    return subset


def save_manifest(samples: List[Dict], path: Path):
    """Save manifest to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"samples": samples, "count": len(samples)}, f, indent=2)
    print(f"Saved manifest: {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data manifests for DeepFilterNet-SE")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Output directory for manifests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--debug-hours",
        type=float,
        default=3.0,
        help="Hours for debug subset",
    )
    args = parser.parse_args()
    
    # =============================================================
    # PRIMARY SPEAKERS: VCTK + LibriSpeech (English)
    # =============================================================
    print("\n=== Indexing PRIMARY speakers (VCTK + LibriSpeech) ===\n")
    
    primary_samples = []
    primary_speakers = set()
    
    # VCTK
    vctk_samples, vctk_speakers = index_vctk(args.data_root)
    primary_samples.extend(vctk_samples)
    primary_speakers.update(vctk_speakers)
    
    # LibriSpeech
    libri_samples, libri_speakers = index_librispeech(args.data_root)
    primary_samples.extend(libri_samples)
    primary_speakers.update(libri_speakers)
    
    print(f"\nPrimary pool: {len(primary_samples)} samples, {len(primary_speakers)} speakers")
    
    # =============================================================
    # BACKGROUND SPEAKERS: DNS4 non-English
    # =============================================================
    print("\n=== Indexing BACKGROUND speakers (DNS4 non-English) ===\n")
    
    background_samples, background_speakers = index_dns4_background(args.data_root)
    print(f"\nBackground pool: {len(background_samples)} samples, {len(background_speakers)} speakers")
    
    # =============================================================
    # NOISE: DNS4 + MUSAN
    # =============================================================
    print("\n=== Indexing NOISE ===\n")
    
    noise_samples = index_dns4_noise(args.data_root)
    musan_samples = index_musan(args.data_root)
    all_noise = noise_samples + musan_samples
    print(f"\nTotal noise: {len(all_noise)} samples")
    
    # =============================================================
    # IMPULSE RESPONSES
    # =============================================================
    print("\n=== Indexing IMPULSE RESPONSES ===\n")
    
    ir_samples = index_impulse_responses(args.data_root)
    
    # =============================================================
    # VALIDATION
    # =============================================================
    if not primary_samples:
        print("\nError: No primary speaker samples found. Please check data paths.")
        return
    
    if not background_samples:
        print("\nWarning: No background speaker samples found. Continuing without background speakers.")
    
    # =============================================================
    # CREATE SPLITS (speaker-disjoint within primary pool)
    # =============================================================
    print("\n=== Creating speaker-disjoint splits ===\n")
    
    train_samples, val_samples, test_samples = create_splits(
        primary_samples, primary_speakers, seed=args.seed
    )
    
    # Create debug subset from training data
    debug_samples = create_debug_subset(train_samples, args.debug_hours, args.seed)
    
    # =============================================================
    # SAVE MANIFESTS
    # =============================================================
    print("\n=== Saving manifests ===\n")
    
    # Primary speaker splits
    save_manifest(train_samples, args.output_dir / "train.json")
    save_manifest(val_samples, args.output_dir / "val.json")
    save_manifest(test_samples, args.output_dir / "test.json")
    save_manifest(debug_samples, args.output_dir / "debug_subset.json")
    
    # Background speakers (single file, used for mixing)
    if background_samples:
        save_manifest(background_samples, args.output_dir / "background.json")
    
    # Noise
    if all_noise:
        save_manifest(all_noise, args.output_dir / "noise.json")
    
    # Impulse responses
    if ir_samples:
        save_manifest(ir_samples, args.output_dir / "impulse_responses.json")
    
    # =============================================================
    # SUMMARY
    # =============================================================
    print("\n" + "="*60)
    print("MANIFEST GENERATION COMPLETE")
    print("="*60)
    print(f"Primary speakers: {len(primary_speakers)} speakers, {len(primary_samples)} samples")
    print(f"Background speakers: {len(background_speakers)} speakers, {len(background_samples)} samples")
    print(f"Noise files: {len(all_noise)} samples")
    print(f"Impulse responses: {len(ir_samples)} samples")
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
