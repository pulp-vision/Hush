#!/usr/bin/env python3
"""Batch offline inference for Advanced-DeepFilterNet checkpoints.

Example:
  python3 scripts/infer_dfnet_batch.py \
    --run-config runs/<run>/run_config.json \
    --checkpoint runs/<run>/checkpoints/model_best.ckpt \
    --input runs/<run>/.../dataset/mixed \
    --output-dir runs/<run>/.../outputs/dfnet_reinfer
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libdf import DF, erb, erb_norm, unit_norm
from model.dfnet_se import DfNetSE, ModelConfig, as_complex, as_real, get_norm_alpha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch inference for Advanced-DeepFilterNet"
    )
    p.add_argument(
        "--run-config", type=Path, required=True, help="Path to run_config.json"
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model checkpoint"
    )
    p.add_argument(
        "--input", type=Path, required=True, help="Input wav file or directory"
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--atten-lim-db", type=float, default=None)
    p.add_argument("--no-delay-compensation", action="store_true")
    p.add_argument(
        "--glob",
        type=str,
        default="*.wav",
        help="Glob pattern when input is a directory",
    )
    p.add_argument(
        "--summary-json", type=Path, default=None, help="Optional summary JSON path"
    )
    p.add_argument(
        "--runtime-csv",
        type=Path,
        default=None,
        help="Optional per-file runtime CSV path",
    )
    return p.parse_args()


def build_model_config(run_cfg: Dict) -> ModelConfig:
    df_c = run_cfg["df"]
    model_c = run_cfg["model"]
    return ModelConfig(
        sr=df_c["sr"],
        fft_size=df_c["fft_size"],
        hop_size=df_c["hop_size"],
        nb_erb=df_c["nb_erb"],
        nb_df=df_c["nb_df"],
        norm_tau=df_c["norm_tau"],
        lsnr_max=df_c["lsnr_max"],
        lsnr_min=df_c["lsnr_min"],
        min_nb_freqs=df_c.get("min_nb_freqs", 2),
        df_order=df_c["df_order"],
        df_lookahead=df_c["df_lookahead"],
        conv_lookahead=model_c["conv_lookahead"],
        conv_ch=model_c["conv_ch"],
        conv_depthwise=model_c["conv_depthwise"],
        convt_depthwise=model_c["convt_depthwise"],
        conv_kernel=tuple(model_c["conv_kernel"]),
        convt_kernel=tuple(model_c["convt_kernel"]),
        conv_kernel_inp=tuple(model_c["conv_kernel_inp"]),
        emb_hidden_dim=model_c["emb_hidden_dim"],
        emb_num_layers=model_c["emb_num_layers"],
        emb_gru_skip_enc=model_c["emb_gru_skip_enc"],
        emb_gru_skip=model_c["emb_gru_skip"],
        df_hidden_dim=model_c["df_hidden_dim"],
        df_gru_skip=model_c["df_gru_skip"],
        df_pathway_kernel_size_t=model_c["df_pathway_kernel_size_t"],
        enc_concat=model_c["enc_concat"],
        df_num_layers=model_c["df_num_layers"],
        df_n_iter=model_c["df_n_iter"],
        lin_groups=model_c["lin_groups"],
        enc_lin_groups=model_c["enc_lin_groups"],
        mask_pf=model_c["mask_pf"],
        pf_beta=model_c["pf_beta"],
        lsnr_dropout=model_c["lsnr_dropout"],
    )


def select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(str(path), backend="soundfile")
    except Exception:
        import soundfile as sf

        wav_np, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(wav_np)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.t()

    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
    return wav


def load_weights(model: DfNetSE, checkpoint: Path, map_location: torch.device) -> None:
    state = torch.load(str(checkpoint), map_location=map_location)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")

    if next(iter(state_dict)).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    try:
        model.model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    if all(k.startswith("model.") for k in state_dict.keys()):
        stripped = {k[len("model.") :]: v for k, v in state_dict.items()}
        model.model.load_state_dict(stripped, strict=True)
        return

    raise RuntimeError(f"Could not load checkpoint weights from {checkpoint}")


def df_features(
    audio: torch.Tensor, df_state: DF, nb_df: int, norm_tau: float, device: torch.device
):
    alpha = get_norm_alpha(df_state.sr(), df_state.hop_size(), norm_tau)
    spec_np = df_state.analysis(audio.numpy(), reset=True)
    erb_fb = df_state.erb_widths()

    spec = torch.as_tensor(spec_np)
    erb_feat_np = erb_norm(erb(spec_np, erb_fb), alpha)
    spec_feat_np = unit_norm(spec_np[..., :nb_df], alpha)

    spec_t = as_real(spec).unsqueeze(1)
    erb_feat_t = torch.as_tensor(erb_feat_np).unsqueeze(1)
    spec_feat_t = as_real(torch.as_tensor(spec_feat_np)).unsqueeze(1)
    return spec_t.to(device), erb_feat_t.to(device), spec_feat_t.to(device)


@torch.no_grad()
def enhance_df_style(
    model: DfNetSE,
    df_state: DF,
    audio: torch.Tensor,
    pad_delay: bool,
    atten_lim_db: float | None,
) -> torch.Tensor:
    orig_len = int(audio.shape[-1])
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()
    if pad_delay:
        audio = torch.nn.functional.pad(audio, (0, n_fft))

    spec, feat_erb, feat_spec = df_features(
        audio=audio,
        df_state=df_state,
        nb_df=model.config.nb_df,
        norm_tau=model.config.norm_tau,
        device=next(model.parameters()).device,
    )

    spec_enh = model.model(spec.clone(), feat_erb, feat_spec)[0]
    spec_enh_c = as_complex(spec_enh.squeeze(1)).cpu()

    if atten_lim_db is not None and abs(float(atten_lim_db)) > 0:
        lim = 10 ** (-abs(float(atten_lim_db)) / 20.0)
        spec_in_c = as_complex(spec.squeeze(1)).cpu()
        spec_enh_c = spec_in_c * lim + spec_enh_c * (1.0 - lim)

    enh_np = df_state.synthesis(spec_enh_c.numpy(), reset=True)
    enh = torch.from_numpy(np.asarray(enh_np, dtype=np.float32))
    if pad_delay:
        delay = n_fft - hop
        enh = enh[:, delay : orig_len + delay]
    return enh


def list_inputs(path: Path, pattern: str) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob(pattern))
        if not files:
            raise ValueError(f"No files matching '{pattern}' in {path}")
        return files
    raise FileNotFoundError(path)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    with args.run_config.resolve().open("r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    cfg = build_model_config(run_cfg)

    device = select_device(args.device)
    print(f"Using device: {device}")

    model = DfNetSE(config=cfg)
    try:
        model = model.to(device)
    except RuntimeError as exc:
        if str(device) == "cuda":
            print(f"CUDA init failed ({exc}). Falling back to CPU.")
            device = torch.device("cpu")
            model = model.to(device)
        else:
            raise

    load_weights(model, args.checkpoint.resolve(), map_location=device)
    model.eval()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_inputs(args.input.resolve(), args.glob)
    rows: List[Dict[str, object]] = []

    for idx, wav_path in enumerate(files, start=1):
        t0 = time.perf_counter()
        status = "ok"
        err = ""
        audio_seconds = 0.0
        out_path = out_dir / wav_path.name
        try:
            wav = load_audio_mono(wav_path, cfg.sr)
            audio_seconds = float(wav.shape[-1]) / float(cfg.sr)
            df_state = DF(
                sr=cfg.sr,
                fft_size=cfg.fft_size,
                hop_size=cfg.hop_size,
                nb_bands=cfg.nb_erb,
                min_nb_erb_freqs=cfg.min_nb_freqs,
            )
            enhanced = enhance_df_style(
                model=model,
                df_state=df_state,
                audio=wav,
                pad_delay=not args.no_delay_compensation,
                atten_lim_db=args.atten_lim_db,
            )
            try:
                torchaudio.save(
                    str(out_path), enhanced.cpu(), cfg.sr, backend="soundfile"
                )
            except Exception:
                import soundfile as sf

                sf.write(str(out_path), enhanced.cpu().t().numpy(), cfg.sr)
        except Exception as exc:
            status = "failed"
            err = str(exc)

        elapsed = time.perf_counter() - t0
        rtf = elapsed / max(audio_seconds, 1e-6)
        rows.append(
            {
                "id": wav_path.stem,
                "input": str(wav_path),
                "output": str(out_path),
                "status": status,
                "error": err,
                "latency_ms": round(elapsed * 1000.0, 4),
                "input_seconds": round(audio_seconds, 6),
                "rtf": round(float(rtf), 6),
            }
        )
        if idx % 10 == 0 or idx == len(files):
            print(f"Processed {idx}/{len(files)}")

    runtime_csv = (
        args.runtime_csv.resolve() if args.runtime_csv else out_dir / "runtime.csv"
    )
    write_csv(runtime_csv, rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    summary = {
        "device": str(device),
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "mean_latency_ms": float(np.mean([float(r["latency_ms"]) for r in ok_rows]))
        if ok_rows
        else None,
        "median_latency_ms": float(np.median([float(r["latency_ms"]) for r in ok_rows]))
        if ok_rows
        else None,
        "mean_rtf": float(np.mean([float(r["rtf"]) for r in ok_rows]))
        if ok_rows
        else None,
    }
    summary_json = (
        args.summary_json.resolve() if args.summary_json else out_dir / "summary.json"
    )
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Inference complete: {summary['n_ok']}/{summary['n_total']} files succeeded")
    print(f"Outputs: {out_dir}")
    print(f"Runtime CSV: {runtime_csv}")
    print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
