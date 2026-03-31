#!/usr/bin/env python3
"""Single-file inference for the Hush (DeepFilterNet-SE) model.

Uses libdf for feature extraction and synthesis, matching the training pipeline
exactly. This produces output identical to the reference denoised samples.

Requirements:
    pip install torch torchaudio soundfile numpy DeepFilterLib

Usage:
    python scripts/infer_single.py \\
        --checkpoint deployment/models/model_best.ckpt \\
        --input noisy.wav \\
        --output enhanced.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libdf import DF, erb, erb_norm, unit_norm
from model.dfnet_se import DfNetSE, as_complex, as_real, get_config, get_norm_alpha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Denoise a single WAV file using Hush.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    p.add_argument("--input", type=Path, required=True, help="Input noisy WAV file")
    p.add_argument("--output", type=Path, required=True, help="Output denoised WAV file")
    p.add_argument(
        "--atten-lim-db",
        type=float,
        default=None,
        help="Max attenuation in dB (default: None = unlimited)",
    )
    p.add_argument(
        "--no-delay-compensation",
        action="store_true",
        help="Skip delay compensation (default: compensate)",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run inference on",
    )
    return p.parse_args()


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio as mono float32 tensor [1, T] at target sample rate."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path), backend="soundfile")
    except Exception:
        import soundfile as sf

        wav_np, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(wav_np)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.t()

    # Convert to mono
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Resample if needed
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    return wav


def select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def enhance(
    model: DfNetSE,
    audio: torch.Tensor,
    pad_delay: bool = True,
    atten_lim_db: float | None = None,
) -> torch.Tensor:
    """Enhance audio using libdf for feature extraction (matches training).

    Parameters
    ----------
    model : DfNetSE
        Loaded model in eval mode.
    audio : torch.Tensor
        Input waveform, shape [1, T], 16 kHz mono float32.
    pad_delay : bool
        If True, compensate for algorithmic delay so output aligns with input.
    atten_lim_db : float or None
        Maximum attenuation in dB. None = unlimited.

    Returns
    -------
    torch.Tensor
        Enhanced waveform, shape [1, T].
    """
    config = model.config
    device = next(model.parameters()).device

    # Create libdf state for feature extraction
    df_state = DF(
        sr=config.sr,
        fft_size=config.fft_size,
        hop_size=config.hop_size,
        nb_bands=config.nb_erb,
        min_nb_erb_freqs=config.min_nb_freqs,
    )

    orig_len = int(audio.shape[-1])
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()

    if pad_delay:
        audio = torch.nn.functional.pad(audio, (0, n_fft))

    # Feature extraction via libdf (same as training)
    alpha = get_norm_alpha(df_state.sr(), df_state.hop_size(), config.norm_tau)
    spec_np = df_state.analysis(audio.numpy(), reset=True)
    erb_fb = df_state.erb_widths()

    spec = torch.as_tensor(spec_np)
    erb_feat_np = erb_norm(erb(spec_np, erb_fb), alpha)
    spec_feat_np = unit_norm(spec_np[..., : config.nb_df], alpha)

    spec_t = as_real(spec).unsqueeze(1).to(device)
    erb_feat_t = torch.as_tensor(erb_feat_np).unsqueeze(1).to(device)
    spec_feat_t = as_real(torch.as_tensor(spec_feat_np)).unsqueeze(1).to(device)

    # Forward pass through model
    spec_enh = model.model(spec_t.clone(), erb_feat_t, spec_feat_t)[0]
    spec_enh_c = as_complex(spec_enh.squeeze(1)).cpu()

    # Apply attenuation limit if specified
    if atten_lim_db is not None and abs(float(atten_lim_db)) > 0:
        lim = 10 ** (-abs(float(atten_lim_db)) / 20.0)
        spec_in_c = as_complex(spec.unsqueeze(1).squeeze(1))
        spec_enh_c = spec_in_c * lim + spec_enh_c * (1.0 - lim)

    # Synthesis via libdf
    enh_np = df_state.synthesis(spec_enh_c.numpy(), reset=True)
    enh = torch.from_numpy(np.asarray(enh_np, dtype=np.float32))

    # Delay compensation
    if pad_delay:
        delay = n_fft - hop
        enh = enh[:, delay : orig_len + delay]

    return enh


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"Device: {device}")

    # Load model
    config = get_config()
    model = DfNetSE(config).to(device)

    state_dict = torch.load(
        str(args.checkpoint.resolve()), map_location=device, weights_only=False
    )
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Handle different key prefixes
    try:
        model.model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            if all(k.startswith("model.") for k in state_dict):
                stripped = {k[6:]: v for k, v in state_dict.items()}
                model.model.load_state_dict(stripped, strict=True)
            else:
                raise RuntimeError(f"Could not load checkpoint: {args.checkpoint}")

    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load audio
    audio = load_audio(args.input.resolve(), config.sr)
    duration = audio.shape[-1] / config.sr
    print(f"Input: {args.input} ({duration:.2f}s, {config.sr} Hz)")

    # Enhance
    enhanced = enhance(
        model,
        audio,
        pad_delay=not args.no_delay_compensation,
        atten_lim_db=args.atten_lim_db,
    )

    # Save
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio

        torchaudio.save(str(args.output.resolve()), enhanced.cpu(), config.sr, backend="soundfile")
    except Exception:
        import soundfile as sf

        sf.write(str(args.output.resolve()), enhanced.squeeze().cpu().numpy(), config.sr)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
