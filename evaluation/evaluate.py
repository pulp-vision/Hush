#!/usr/bin/env python3
"""
Evaluation script for DeepFilterNet-SE.

Computes metrics on test set: SI-SDR, SDR, PESQ, STOI.

Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/best.pt --test-manifest data/manifests/test.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from training.loss import si_sdr


def compute_sdr(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute Signal-to-Distortion Ratio.
    
    Args:
        estimate: [B, T] or [B, 1, T]
        target: [B, T] or [B, 1, T]
    
    Returns:
        SDR in dB [B]
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    noise = estimate - target
    sdr = 10 * torch.log10(
        torch.sum(target ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps
    )
    return sdr


def compute_pesq(estimate: np.ndarray, target: np.ndarray, sample_rate: int = 16000) -> float:
    """Compute PESQ score."""
    try:
        from pesq import pesq
        # PESQ expects 1D arrays
        return pesq(sample_rate, target.flatten(), estimate.flatten(), "wb")
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        return 0.0


def compute_stoi(estimate: np.ndarray, target: np.ndarray, sample_rate: int = 16000) -> float:
    """Compute STOI score."""
    try:
        from pystoi import stoi
        return stoi(target.flatten(), estimate.flatten(), sample_rate, extended=False)
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return 0.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_manifest: Path,
    device: torch.device,
    sample_rate: int = 16000,
    segment_length: float = 5.0,
    compute_intrusive: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_manifest: Path to test manifest JSON
        device: Computation device
        sample_rate: Sample rate
        segment_length: Segment length for evaluation
        compute_intrusive: Whether to compute PESQ/STOI (slow)
    
    Returns:
        Dictionary with average metrics
    """
    from data.augmentation import MixingAugmentor
    from data.dataset import SpeakerMixtureDataset
    
    model.eval()
    
    # Create test dataset
    augmentor = MixingAugmentor(
        sample_rate=sample_rate,
        secondary_scale_range=(0.4, 0.4),  # Fixed for reproducibility
        pitch_shift_range=(-5.0, -5.0),
        noise_snr_range=(10.0, 10.0),
        p_secondary=1.0,
        p_noise=1.0,
    )
    
    dataset = SpeakerMixtureDataset(
        manifest_path=test_manifest,
        sample_rate=sample_rate,
        segment_length=segment_length,
        augmentor=augmentor,
        return_metadata=True,
    )
    
    metrics = {
        "si_sdr": [],
        "si_sdri": [],  # Improvement over mixture
        "sdr": [],
        "sdri": [],
        "pesq": [],
        "stoi": [],
    }
    
    for idx in tqdm(range(min(len(dataset), 500)), desc="Evaluating"):  # Limit samples
        mixture, target, metadata = dataset[idx]
        
        # Add batch dimension
        mixture = mixture.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        # Enhance
        enhanced = model(mixture)
        
        # Compute metrics
        sisdr_enh = si_sdr(enhanced, target).item()
        sisdr_mix = si_sdr(mixture, target).item()
        
        sdr_enh = compute_sdr(enhanced, target).item()
        sdr_mix = compute_sdr(mixture, target).item()
        
        metrics["si_sdr"].append(sisdr_enh)
        metrics["si_sdri"].append(sisdr_enh - sisdr_mix)
        metrics["sdr"].append(sdr_enh)
        metrics["sdri"].append(sdr_enh - sdr_mix)
        
        # PESQ/STOI (slow, optional)
        if compute_intrusive:
            enh_np = enhanced.squeeze().cpu().numpy()
            tgt_np = target.squeeze().cpu().numpy()
            
            metrics["pesq"].append(compute_pesq(enh_np, tgt_np, sample_rate))
            metrics["stoi"].append(compute_stoi(enh_np, tgt_np, sample_rate))
    
    # Compute averages
    results = {}
    for key, values in metrics.items():
        if values:
            results[key] = float(np.mean(values))
            results[f"{key}_std"] = float(np.std(values))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepFilterNet-SE")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        required=True,
        help="Path to test manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--no-intrusive",
        action="store_true",
        help="Skip PESQ/STOI computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    
    # Load model
    from export.onnx_export import load_model
    model, config = load_model(args.checkpoint, args.device)
    model = model.to(args.device)
    
    # Evaluate
    results = evaluate(
        model,
        args.test_manifest,
        torch.device(args.device),
        sample_rate=config.model.stft.sample_rate,
        compute_intrusive=not args.no_intrusive,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for key, value in results.items():
        if not key.endswith("_std"):
            std = results.get(f"{key}_std", 0)
            print(f"{key:>12}: {value:7.3f} ± {std:.3f}")
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
