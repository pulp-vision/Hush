#!/usr/bin/env python3
"""Debug script to diagnose training issues."""

import sys
sys.path.insert(0, '/home/ubuntu/Noise-Cancellation')

import torch
import json
from pathlib import Path

from data.augmentation import load_audio, MixingAugmentor, random_crop
from model.dfnet_se import DfNetSE
from training.loss import si_sdr, CombinedLoss

def main():
    print("=" * 60)
    print("TRAINING DIAGNOSTICS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model
    print("\n1. Loading model...")
    model = DfNetSE().to(device)
    model.eval()
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load a sample
    print("\n2. Loading sample data...")
    manifest_dir = Path("/home/ubuntu/Noise-Cancellation/data/manifests")
    with open(manifest_dir / "train.json") as f:
        train = json.load(f)
    with open(manifest_dir / "noise.json") as f:
        noise = json.load(f)
    
    sample = train["samples"][0]
    noise_sample = noise["samples"][0]
    
    print(f"   Primary: {sample['path']}")
    print(f"   Noise: {noise_sample['path']}")
    
    # Load audio
    primary = load_audio(sample["path"], 16000)
    noise_audio = load_audio(noise_sample["path"], 16000)
    
    # Crop
    seg_len = 48000  # 3 seconds
    primary = random_crop(primary, seg_len)
    noise_audio = random_crop(noise_audio, seg_len)
    
    print(f"   Primary shape: {primary.shape}")
    print(f"   Noise shape: {noise_audio.shape}")
    
    # Create mixture
    print("\n3. Creating mixture...")
    augmentor = MixingAugmentor(sample_rate=16000)
    
    # Get a secondary speaker
    secondary_sample = train["samples"][100]  # Different speaker
    secondary = load_audio(secondary_sample["path"], 16000)
    secondary = random_crop(secondary, seg_len)
    
    mixture, clean_target = augmentor(primary, secondary, noise_audio)
    
    print(f"   Mixture shape: {mixture.shape}")
    print(f"   Clean target shape: {clean_target.shape}")
    print(f"   Mixture RMS: {mixture.pow(2).mean().sqrt().item():.4f}")
    print(f"   Target RMS: {clean_target.pow(2).mean().sqrt().item():.4f}")
    
    # Check if mixture == target (would be a bug)
    diff = (mixture - clean_target).abs().mean().item()
    print(f"   |Mixture - Target|: {diff:.6f}")
    if diff < 0.001:
        print("   ⚠️ WARNING: Mixture and target are nearly identical!")
    
    # Run model
    print("\n4. Running model inference...")
    mixture_batch = mixture.unsqueeze(0).to(device)  # [1, 1, T]
    target_batch = clean_target.unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced = model(mixture_batch)
    
    print(f"   Enhanced shape: {enhanced.shape}")
    print(f"   Enhanced RMS: {enhanced.pow(2).mean().sqrt().item():.4f}")
    
    # Check if model is just passing through input
    diff_from_mixture = (enhanced - mixture_batch).abs().mean().item()
    diff_from_target = (enhanced - target_batch).abs().mean().item()
    print(f"   |Enhanced - Mixture|: {diff_from_mixture:.6f}")
    print(f"   |Enhanced - Target|: {diff_from_target:.6f}")
    
    if diff_from_mixture < 0.01:
        print("   ⚠️ WARNING: Model output is nearly identical to input!")
    
    # Compute SI-SDR
    print("\n5. Computing metrics...")
    sisdr_enhanced = si_sdr(enhanced, target_batch).item()
    sisdr_mixture = si_sdr(mixture_batch, target_batch).item()
    
    print(f"   SI-SDR(mixture, target): {sisdr_mixture:.2f} dB (baseline)")
    print(f"   SI-SDR(enhanced, target): {sisdr_enhanced:.2f} dB (model output)")
    print(f"   Improvement: {sisdr_enhanced - sisdr_mixture:.2f} dB")
    
    # Compute loss
    print("\n6. Computing loss...")
    criterion = CombinedLoss()
    loss, loss_dict = criterion(enhanced, target_batch, mixture_batch)
    print(f"   Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")
    
    # Check for NaN/Inf
    print("\n7. Checking for numerical issues...")
    if torch.isnan(enhanced).any():
        print("   ⚠️ WARNING: Enhanced output contains NaN!")
    if torch.isinf(enhanced).any():
        print("   ⚠️ WARNING: Enhanced output contains Inf!")
    if torch.isnan(loss):
        print("   ⚠️ WARNING: Loss is NaN!")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    issues = []
    if diff < 0.001:
        issues.append("Mixture == Target (data bug)")
    if diff_from_mixture < 0.01:
        issues.append("Model is passing through input")
    if sisdr_enhanced < sisdr_mixture:
        issues.append("Model making audio WORSE")
    if torch.isnan(enhanced).any() or torch.isnan(loss):
        issues.append("NaN values detected")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("✅ No obvious issues detected")
        print(f"   Model is producing output different from input")
        print(f"   Early training SI-SDR of {sisdr_enhanced:.1f}dB is normal for untrained model")

if __name__ == "__main__":
    main()
