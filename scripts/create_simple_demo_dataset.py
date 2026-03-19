#!/usr/bin/env python3
"""Create a small, audible demo dataset with primary/background/noise mixtures.

Outputs:
  - clean/: primary-only target
  - mixed/: primary + background speaker
  - noisy/: primary + background speaker + noise
  - cleaned/: model-enhanced output from noisy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch


def read_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Manifest has no samples: {path}")
    return samples


def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # soundfile returns [T, C]
    audio = audio.mean(axis=1)
    if sr == target_sr:
        return audio
    src_len = len(audio)
    dst_len = max(1, int(round(src_len * target_sr / sr)))
    src = np.linspace(0.0, 1.0, src_len, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, dst_len, dtype=np.float64)
    return np.interp(dst, src, audio).astype(np.float32)


def fit_to_len(x: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        start = rng.randint(0, len(x) - n)
        return x[start : start + n]
    # tile short signals
    reps = int(math.ceil(n / max(1, len(x))))
    y = np.tile(x, reps)
    return y[:n]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def scale_for_snr(ref: np.ndarray, sig: np.ndarray, snr_db: float) -> np.ndarray:
    ref_rms = rms(ref)
    sig_rms = rms(sig)
    if sig_rms < 1e-9:
        return np.zeros_like(sig)
    target_sig_rms = ref_rms / (10.0 ** (snr_db / 20.0))
    return sig * (target_sig_rms / sig_rms)


def peak_normalize(*signals: np.ndarray, peak: float = 0.95) -> List[np.ndarray]:
    max_abs = max(float(np.max(np.abs(s))) for s in signals)
    if max_abs < 1e-9:
        return [s.copy() for s in signals]
    g = min(1.0, peak / max_abs)
    return [s * g for s in signals]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create small audible demo dataset")
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/Noise-Cancellation"))
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--background-manifest", type=Path, required=True)
    p.add_argument("--noise-manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-config", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--snr-bg-min", type=float, default=3.0)
    p.add_argument("--snr-bg-max", type=float, default=8.0)
    p.add_argument("--snr-noise-min", type=float, default=18.0)
    p.add_argument("--snr-noise-max", type=float, default=28.0)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _resolve_run_config_path(checkpoint_path: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit.resolve()
    candidates = [
        checkpoint_path.parent / "run_config.json",
        checkpoint_path.parent.parent / "run_config.json",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_clean = out_dir / "clean"
    out_mixed = out_dir / "mixed"
    out_noisy = out_dir / "noisy"
    out_cleaned = out_dir / "cleaned"
    out_bg = out_dir / "background"
    for d in [out_clean, out_mixed, out_noisy, out_cleaned, out_bg]:
        d.mkdir(parents=True, exist_ok=True)

    train_samples = read_manifest(args.train_manifest.resolve())
    bg_samples = read_manifest(args.background_manifest.resolve())
    noise_samples = read_manifest(args.noise_manifest.resolve())

    adv_root = root / "Advanced DeepFilterNet"
    if str(adv_root) not in sys.path:
        sys.path.insert(0, str(adv_root))
    from model.dfnet_se import DfNetSE, get_config  # type: ignore

    cfg = get_config()
    run_cfg_path = _resolve_run_config_path(args.checkpoint.resolve(), args.run_config)
    if run_cfg_path is not None:
        with run_cfg_path.open("r", encoding="utf-8") as f:
            run_cfg = json.load(f)
        for key, val in run_cfg.get("df", {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        for key, val in run_cfg.get("model", {}).items():
            if hasattr(cfg, key):
                if key in {"conv_kernel", "convt_kernel", "conv_kernel_inp"} and isinstance(val, list):
                    val = tuple(val)
                setattr(cfg, key, val)
        print(f"Loaded model config from {run_cfg_path}")

    model = DfNetSE(cfg)
    state = torch.load(args.checkpoint.resolve(), map_location="cpu")
    model.model.load_state_dict(state, strict=True)
    model.eval()

    n = int(args.seconds * args.sr)
    rows: List[Dict[str, object]] = []

    for i in range(args.n_samples):
        pri = rng.choice(train_samples)
        bkg = rng.choice(bg_samples)
        noi = rng.choice(noise_samples)

        pri_path = (root / str(pri["path"])).resolve()
        bkg_path = (root / str(bkg["path"])).resolve()
        noi_path = (root / str(noi["path"])).resolve()

        primary = load_audio_mono(pri_path, args.sr)
        background = load_audio_mono(bkg_path, args.sr)
        noise = load_audio_mono(noi_path, args.sr)

        primary = fit_to_len(primary, n, rng)
        background = fit_to_len(background, n, rng)
        noise = fit_to_len(noise, n, rng)

        snr_bg = rng.uniform(args.snr_bg_min, args.snr_bg_max)
        snr_noise = rng.uniform(args.snr_noise_min, args.snr_noise_max)

        background_scaled = scale_for_snr(primary, background, snr_bg)
        mixed = primary + background_scaled

        noise_scaled = scale_for_snr(mixed, noise, snr_noise)
        noisy = mixed + noise_scaled

        primary, background_scaled, mixed, noisy = peak_normalize(
            primary, background_scaled, mixed, noisy
        )

        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy).float().view(1, 1, -1)
            cleaned = model(noisy_t).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        uid = f"sample_{i:03d}"
        sf.write(str(out_clean / f"{uid}.wav"), primary, args.sr)
        sf.write(str(out_bg / f"{uid}.wav"), background_scaled, args.sr)
        sf.write(str(out_mixed / f"{uid}.wav"), mixed, args.sr)
        sf.write(str(out_noisy / f"{uid}.wav"), noisy, args.sr)
        sf.write(str(out_cleaned / f"{uid}.wav"), cleaned, args.sr)

        rows.append(
            {
                "id": uid,
                "primary_path": str(pri_path),
                "background_path": str(bkg_path),
                "noise_path": str(noi_path),
                "snr_bg_db": round(float(snr_bg), 3),
                "snr_noise_db": round(float(snr_noise), 3),
            }
        )

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sr": args.sr,
                "seconds": args.seconds,
                "n_samples": args.n_samples,
                "snr_bg_range_db": [args.snr_bg_min, args.snr_bg_max],
                "snr_noise_range_db": [args.snr_noise_min, args.snr_noise_max],
                "samples": rows,
            },
            f,
            indent=2,
        )

    print(f"Wrote demo dataset to: {out_dir}")


if __name__ == "__main__":
    main()
