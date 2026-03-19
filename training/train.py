#!/usr/bin/env python3
"""
Training script for DeepFilterNet3-SE.

Closely mirrors the original df/train.py training loop with these differences:
  1. Uses our training/config.py dataclass config instead of df.config INI files
  2. Adds separation head forward/loss
  3. 16 kHz sample rate

Usage (libdfdata HDF5 mode):
    python training/train.py \\
        --data-config data/hdf5/dataset.cfg \\
        --data-dir data/hdf5 \\
        --base-dir runs/exp1

Usage (debug, CPU):
    python training/train.py \\
        --data-config data/hdf5/dataset.cfg \\
        --data-dir data/hdf5 \\
        --base-dir runs/debug \\
        --device cpu --debug --max-steps 5
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import site
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def _reexec_with_pytorch_cuda_libs() -> None:
    """Re-exec process with bundled NVIDIA libs to avoid cuDNN mismatches."""
    if os.environ.get("NC_TORCH_LIBS_READY") == "1":
        return

    cudnn_dir: Optional[str] = None
    cublas_dir: Optional[str] = None
    for base in [site.getusersitepackages(), *site.getsitepackages()]:
        c1 = os.path.join(base, "nvidia", "cudnn", "lib")
        c2 = os.path.join(base, "nvidia", "cublas", "lib")
        if cudnn_dir is None and os.path.isdir(c1):
            cudnn_dir = c1
        if cublas_dir is None and os.path.isdir(c2):
            cublas_dir = c2

    if cudnn_dir is None or cublas_dir is None:
        return

    env = os.environ.copy()
    # Prepend bundled libs but keep existing runtime paths (e.g. libcuda, cudart).
    # Replacing LD_LIBRARY_PATH entirely can break CUDA device discovery.
    existing = env.get("LD_LIBRARY_PATH", "")
    lib_paths = [cudnn_dir, cublas_dir]
    if existing:
        lib_paths.extend([p for p in existing.split(":") if p])
    # Keep order-stable unique paths.
    dedup = list(dict.fromkeys(lib_paths))
    env["LD_LIBRARY_PATH"] = ":".join(dedup)
    env["NC_TORCH_LIBS_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_reexec_with_pytorch_cuda_libs()

import numpy as np
import torch
import torchaudio
from torch import Tensor, nn, optim
from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.autograd.grad_mode import set_grad_enabled
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.types import Number
from torch.utils.tensorboard import SummaryWriter

# Allow direct execution via `python training/train.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.dfnet_se import (
    DfNet, ModelConfig, compute_erb_widths, erb_fb,
    ExponentialUnitNorm, as_complex, as_real, get_config,
)
from training.config import Config, get_default_config, load_config_from_ini
from training.loss import Istft, Loss
from training.lr import cosine_scheduler

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

try:
    from pesq import pesq as pesq_fn

    HAS_PESQ = True
except ImportError:
    pesq_fn = None
    HAS_PESQ = False

try:
    from pystoi import stoi as stoi_fn

    HAS_STOI = True
except ImportError:
    stoi_fn = None
    HAS_STOI = False

# Rust data loader
try:
    from libdfdata import PytorchDataLoader as DataLoader
    from libdf import DF
    HAS_LIBDF = True
except ImportError:
    HAS_LIBDF = False
    DataLoader = None
    DF = None

should_stop = False
debug = False
MAX_NANS = 50


def _snr_db(clean: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    clean = clean.astype(np.float32, copy=False)
    estimate = estimate.astype(np.float32, copy=False)
    noise = estimate - clean
    s_pow = float(np.mean(clean * clean) + eps)
    n_pow = float(np.mean(noise * noise) + eps)
    return 10.0 * math.log10(s_pow / n_pow)


def _si_sdr_db(clean: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    clean = clean.astype(np.float32, copy=False)
    estimate = estimate.astype(np.float32, copy=False)
    clean = clean - float(np.mean(clean))
    estimate = estimate - float(np.mean(estimate))
    denom = float(np.dot(clean, clean) + eps)
    alpha = float(np.dot(estimate, clean) / denom)
    proj = alpha * clean
    noise = estimate - proj
    proj_pow = float(np.dot(proj, proj) + eps)
    noise_pow = float(np.dot(noise, noise) + eps)
    return 10.0 * math.log10(proj_pow / noise_pow)


class ValidationMetricTracker:
    """Compute objective speech metrics on a bounded validation subset."""

    def __init__(
        self,
        sr: int,
        max_batches: int,
        max_samples: int,
        enable_stoi: bool = True,
        enable_pesq: bool = True,
    ):
        self.sr = int(sr)
        self.max_batches = max(0, int(max_batches))
        self.max_samples = max(0, int(max_samples))
        self.enable_stoi = bool(enable_stoi) and HAS_STOI
        self.enable_pesq = bool(enable_pesq) and HAS_PESQ and self.sr in (8000, 16000)
        self._batches_seen = 0
        self._samples_seen = 0
        self._store: Dict[str, List[float]] = {
            "SI_SDR": [],
            "Noisy_SI_SDR": [],
            "SI_SDRi": [],
            "SNR": [],
            "Noisy_SNR": [],
            "SNRi": [],
            "STOI": [],
            "Noisy_STOI": [],
            "ESTOI": [],
            "Noisy_ESTOI": [],
            "PESQ_WB": [],
            "Noisy_PESQ_WB": [],
        }

    def should_update(self) -> bool:
        if self.max_batches == 0 or self.max_samples == 0:
            return False
        if self._batches_seen >= self.max_batches:
            return False
        if self._samples_seen >= self.max_samples:
            return False
        return True

    @torch.no_grad()
    def update(self, clean_spec: Tensor, noisy_spec: Tensor, enhanced_spec: Tensor, istft_fn: nn.Module) -> None:
        if not self.should_update():
            return
        self._batches_seen += 1

        clean_td = istft_fn(clean_spec).detach().cpu().numpy()
        noisy_td = istft_fn(noisy_spec).detach().cpu().numpy()
        enh_td = istft_fn(enhanced_spec).detach().cpu().numpy()

        remaining = self.max_samples - self._samples_seen
        n_batch = int(min(clean_td.shape[0], remaining))
        pesq_mode = "wb" if self.sr == 16000 else "nb"

        for i in range(n_batch):
            clean = np.asarray(clean_td[i], dtype=np.float32).reshape(-1)
            noisy = np.asarray(noisy_td[i], dtype=np.float32).reshape(-1)
            enh = np.asarray(enh_td[i], dtype=np.float32).reshape(-1)
            n = min(clean.shape[0], noisy.shape[0], enh.shape[0])
            if n < 32:
                continue
            clean = np.clip(clean[:n], -1.0, 1.0)
            noisy = np.clip(noisy[:n], -1.0, 1.0)
            enh = np.clip(enh[:n], -1.0, 1.0)

            if not np.isfinite(clean).all() or not np.isfinite(noisy).all() or not np.isfinite(enh).all():
                continue
            if float(np.mean(clean * clean)) < 1e-12:
                continue

            try:
                noisy_si_sdr = _si_sdr_db(clean, noisy)
                enh_si_sdr = _si_sdr_db(clean, enh)
                self._store["Noisy_SI_SDR"].append(noisy_si_sdr)
                self._store["SI_SDR"].append(enh_si_sdr)
                self._store["SI_SDRi"].append(enh_si_sdr - noisy_si_sdr)

                noisy_snr = _snr_db(clean, noisy)
                enh_snr = _snr_db(clean, enh)
                self._store["Noisy_SNR"].append(noisy_snr)
                self._store["SNR"].append(enh_snr)
                self._store["SNRi"].append(enh_snr - noisy_snr)
            except Exception:
                pass

            if self.enable_stoi and stoi_fn is not None:
                try:
                    noisy_stoi = float(stoi_fn(clean, noisy, self.sr, extended=False))
                    enh_stoi = float(stoi_fn(clean, enh, self.sr, extended=False))
                    self._store["Noisy_STOI"].append(noisy_stoi)
                    self._store["STOI"].append(enh_stoi)
                except Exception:
                    pass
                try:
                    noisy_estoi = float(stoi_fn(clean, noisy, self.sr, extended=True))
                    enh_estoi = float(stoi_fn(clean, enh, self.sr, extended=True))
                    self._store["Noisy_ESTOI"].append(noisy_estoi)
                    self._store["ESTOI"].append(enh_estoi)
                except Exception:
                    pass

            if self.enable_pesq and pesq_fn is not None:
                try:
                    noisy_pesq = float(pesq_fn(self.sr, clean, noisy, pesq_mode))
                    enh_pesq = float(pesq_fn(self.sr, clean, enh, pesq_mode))
                    self._store["Noisy_PESQ_WB"].append(noisy_pesq)
                    self._store["PESQ_WB"].append(enh_pesq)
                except Exception:
                    pass

            self._samples_seen += 1
            if self._samples_seen >= self.max_samples:
                break

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "MetricSamples": float(self._samples_seen),
            "MetricBatches": float(self._batches_seen),
        }
        for key, values in self._store.items():
            if values:
                out[key] = float(np.mean(values))
        return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_norm_alpha(sr: int, hop_size: int, tau: float) -> float:
    return math.exp(-hop_size / sr / tau)


RUN_CONFIG_NAME = "run_config.json"
MLFLOW_RUN_ID_NAME = ".mlflow_run_id"
DF_COMPAT_FIELDS = (
    "sr",
    "fft_size",
    "hop_size",
    "nb_erb",
    "nb_df",
    "df_order",
    "df_lookahead",
    "min_nb_freqs",
)
MODEL_COMPAT_FIELDS = (
    "conv_ch",
    "emb_hidden_dim",
    "emb_num_layers",
    "df_hidden_dim",
    "df_num_layers",
)


def _normalize_cfg_value(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def _read_saved_run_config(base_dir: str, cp_dir: str) -> Tuple[Optional[Dict], Optional[str]]:
    candidates = [
        os.path.join(cp_dir, RUN_CONFIG_NAME),
        os.path.join(base_dir, RUN_CONFIG_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
    return None, None


def _assert_run_config_compatible(config: Config, saved_cfg: Dict, saved_path: str) -> None:
    current = asdict(config)
    cur_df = current.get("df", {})
    cur_model = current.get("model", {})
    old_df = saved_cfg.get("df", {}) if isinstance(saved_cfg, dict) else {}
    old_model = saved_cfg.get("model", {}) if isinstance(saved_cfg, dict) else {}

    mismatches: List[str] = []
    for k in DF_COMPAT_FIELDS:
        if _normalize_cfg_value(old_df.get(k)) != _normalize_cfg_value(cur_df.get(k)):
            mismatches.append(f"df.{k}: saved={old_df.get(k)} current={cur_df.get(k)}")
    for k in MODEL_COMPAT_FIELDS:
        if _normalize_cfg_value(old_model.get(k)) != _normalize_cfg_value(cur_model.get(k)):
            mismatches.append(f"model.{k}: saved={old_model.get(k)} current={cur_model.get(k)}")

    if mismatches:
        msg = "\n  - " + "\n  - ".join(mismatches)
        raise RuntimeError(
            "Run config is incompatible with this checkpoint directory.\n"
            f"Saved config: {saved_path}\n"
            "Mismatched fields:" + msg + "\n"
            "Use the original run config (especially fft_size/hop_size/sr) or a new --base-dir."
        )


def _write_run_config(config: Config, base_dir: str, cp_dir: str) -> None:
    payload = asdict(config)
    for path in [os.path.join(base_dir, RUN_CONFIG_NAME), os.path.join(cp_dir, RUN_CONFIG_NAME)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _flatten_cfg(d: Dict, parent: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_cfg(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = str(v)
    return out


def _read_mlflow_run_id(base_dir: str) -> Optional[str]:
    path = os.path.join(base_dir, MLFLOW_RUN_ID_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        rid = f.read().strip()
    return rid or None


def _write_mlflow_run_id(base_dir: str, run_id: str) -> None:
    path = os.path.join(base_dir, MLFLOW_RUN_ID_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(run_id)


def _safe_mlflow_log_metrics(prefix: str, metrics: Dict, step: int) -> None:
    if not HAS_MLFLOW or mlflow is None:
        return
    payload: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            payload[f"{prefix}/{k}"] = float(v)
    if payload:
        mlflow.log_metrics(payload, step=step)


def _setup_mlflow(config: Config, args, checkpoint_dir: str):
    mc = getattr(config, "mlflow", None)
    if mc is None:
        return None
    if not mc.enabled:
        return None
    if not HAS_MLFLOW or mlflow is None:
        raise RuntimeError(
            "MLflow tracking is enabled but mlflow is not installed. "
            "Install it with: python3 -m pip install mlflow"
        )

    tracking_uri = mc.tracking_uri.strip() if mc.tracking_uri else ""
    if not tracking_uri:
        tracking_uri = f"file:{os.path.join(args.base_dir, 'mlruns')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mc.experiment_name)

    run_id = _read_mlflow_run_id(args.base_dir) if args.resume else None
    run_name = mc.run_name.strip() if mc.run_name else ""
    if run_id:
        run = mlflow.start_run(run_id=run_id)
        print(f"MLflow resumed run_id={run_id}")
    else:
        run = mlflow.start_run(run_name=run_name or Path(args.base_dir).name)
        run_id = run.info.run_id
        _write_mlflow_run_id(args.base_dir, run_id)
        print(f"MLflow started run_id={run_id}")

    mlflow.set_tags(
        {
            "project": "Advanced DeepFilterNet",
            "base_dir": args.base_dir,
            "data_dir": args.data_dir,
            "data_config": args.data_config,
        }
    )
    flat_cfg = _flatten_cfg(asdict(config))
    mlflow.log_params(flat_cfg)

    run_cfg_path = os.path.join(args.base_dir, RUN_CONFIG_NAME)
    if os.path.exists(run_cfg_path):
        mlflow.log_artifact(run_cfg_path, artifact_path="config")
    if os.path.exists(args.data_config):
        mlflow.log_artifact(args.data_config, artifact_path="config")
    cache_cfg = os.path.join(os.path.dirname(args.data_config), ".cache_dataset.cfg")
    if os.path.exists(cache_cfg):
        mlflow.log_artifact(cache_cfg, artifact_path="config")
    return run


def _finalize_mlflow(config: Config, checkpoint_dir: str, status: str = "FINISHED") -> None:
    if not HAS_MLFLOW or mlflow is None or mlflow.active_run() is None:
        return
    mc = getattr(config, "mlflow", None)
    if mc is not None and mc.log_artifacts:
        for name in ("model_best.ckpt", "model_latest.ckpt", "opt_latest.ckpt", ".best_metric"):
            p = os.path.join(checkpoint_dir, name)
            if os.path.exists(p):
                mlflow.log_artifact(p, artifact_path="checkpoints")
    mlflow.end_run(status=status)


def save_checkpoint(obj, name: str, cp_dir: str, epoch: int,
                    metric=None, cmp="min"):
    """Save model or optimizer checkpoint."""
    path = os.path.join(cp_dir, f"{name}_{epoch}.ckpt")
    torch.save(obj.state_dict(), path)
    # Also save as 'latest'
    latest = os.path.join(cp_dir, f"{name}_latest.ckpt")
    torch.save(obj.state_dict(), latest)
    # Save best if metric provided
    if metric is not None:
        best_path = os.path.join(cp_dir, f"{name}_best.ckpt")
        marker = os.path.join(cp_dir, ".best_metric")
        prev_best = None
        if os.path.exists(marker):
            with open(marker) as f:
                try:
                    prev_best = float(f.read().strip())
                except ValueError:
                    pass
        is_better = prev_best is None
        if prev_best is not None:
            is_better = metric < prev_best if cmp == "min" else metric > prev_best
        if is_better:
            torch.save(obj.state_dict(), best_path)
            with open(marker, "w") as f:
                f.write(str(metric))


def load_checkpoint(obj, name: str, cp_dir: str, epoch: str = "latest"):
    """Load model or optimizer checkpoint."""
    if epoch == "latest":
        path = os.path.join(cp_dir, f"{name}_latest.ckpt")
    elif epoch == "best":
        path = os.path.join(cp_dir, f"{name}_best.ckpt")
    else:
        path = os.path.join(cp_dir, f"{name}_{epoch}.ckpt")
    if os.path.exists(path):
        # These checkpoints are local training artifacts; allow full unpickling
        # for optimizer state compatibility across PyTorch versions.
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        obj.load_state_dict(state)
        print(f"Loaded {name} from {path}")
        return True
    return False


def get_start_epoch(cp_dir: str) -> int:
    """Find the latest epoch from checkpoint files."""
    if not os.path.isdir(cp_dir):
        return 0
    max_epoch = 0
    for f in os.listdir(cp_dir):
        if f.startswith("model_") and f.endswith(".ckpt") and "latest" not in f and "best" not in f:
            try:
                e = int(f.replace("model_", "").replace(".ckpt", ""))
                max_epoch = max(max_epoch, e)
            except ValueError:
                pass
    return max_epoch


def check_patience(cp_dir: str, max_patience: int, new_metric: float,
                   cmp: str = "min") -> bool:
    """Check early stopping patience."""
    patience_file = os.path.join(cp_dir, ".patience")
    if os.path.exists(patience_file):
        with open(patience_file) as f:
            data = f.read().strip().split(",")
            counter = int(data[0])
            best = float(data[1])
    else:
        counter = 0
        best = float("inf") if cmp == "min" else float("-inf")

    is_better = new_metric < best if cmp == "min" else new_metric > best
    if is_better:
        counter = 0
        best = new_metric
    else:
        counter += 1

    with open(patience_file, "w") as f:
        f.write(f"{counter},{best}")

    return counter < max_patience


def detach_hidden(model: nn.Module):
    """Detach hidden states of GRU modules."""
    for m in model.modules():
        if hasattr(m, "gru") and isinstance(m.gru, nn.GRU):
            m.gru.flatten_parameters()


def _as_complex_spec(x: Tensor) -> Tensor:
    """Convert loader output to complex STFT tensor when needed."""
    if torch.is_complex(x):
        return x
    if x.ndim >= 1 and x.shape[-1] == 2 and x.is_floating_point():
        return torch.view_as_complex(x.contiguous())
    return x.to(torch.complex64)


@torch.no_grad()
def run_data_contract_preflight(
    loader,
    seed: int,
    n_batches: int,
    p_interfer_sp: float,
    min_primary_dominance_db: float,
) -> None:
    """Fail fast on data-contract issues before starting long training."""
    n_batches = max(1, int(n_batches))
    checked_batches = 0
    checked_samples = 0
    active_interferer_samples = 0
    dominance_ratios: List[Tensor] = []

    for i, batch in enumerate(loader.iter_epoch("train", seed)):
        if i >= n_batches:
            break
        if not hasattr(batch, "interferer") or batch.interferer is None:
            raise RuntimeError(
                "Strict data-contract failed: dataloader batch is missing 'interferer'. "
                "Rebuild/reinstall libdfdata and ensure Advanced loader is active."
            )

        speech = _as_complex_spec(batch.speech.detach())
        interferer = _as_complex_spec(batch.interferer.detach())
        if speech.shape != interferer.shape:
            raise RuntimeError(
                f"Strict data-contract failed: shape mismatch speech={tuple(speech.shape)} "
                f"interferer={tuple(interferer.shape)}"
            )
        if not torch.isfinite(speech.real).all() or not torch.isfinite(interferer.real).all():
            raise RuntimeError("Strict data-contract failed: non-finite values in speech/interferer.")

        dims = tuple(range(1, speech.ndim))
        speech_pow = speech.abs().pow(2).mean(dim=dims)
        interferer_pow = interferer.abs().pow(2).mean(dim=dims)
        active_mask = interferer_pow > 1e-12
        if active_mask.any():
            active_interferer_samples += int(active_mask.sum().item())
            ratio_db = 10.0 * torch.log10(
                (speech_pow[active_mask] + 1e-12) / (interferer_pow[active_mask] + 1e-12)
            )
            dominance_ratios.append(ratio_db.cpu())

        checked_batches += 1
        checked_samples += int(speech.shape[0])

    if checked_batches == 0 or checked_samples == 0:
        raise RuntimeError("Strict data-contract failed: could not sample any training batch.")

    active_rate = active_interferer_samples / float(checked_samples)
    print(
        "Data-contract preflight: "
        f"batches={checked_batches}, samples={checked_samples}, "
        f"active_interferer_samples={active_interferer_samples} ({active_rate:.2%})"
    )

    if p_interfer_sp > 0 and active_interferer_samples == 0:
        raise RuntimeError(
            "Strict data-contract failed: no active interferer samples were observed "
            f"in {checked_batches} train batches while p_interfer_sp={p_interfer_sp}."
        )

    if dominance_ratios:
        ratios = torch.cat(dominance_ratios)
        min_db = float(ratios.min().item())
        p10_db = float(torch.quantile(ratios, 0.10).item())
        med_db = float(ratios.median().item())
        print(
            "Primary-vs-interferer dominance (dB): "
            f"min={min_db:.2f}, p10={p10_db:.2f}, median={med_db:.2f}"
        )
        if min_db < min_primary_dominance_db:
            raise RuntimeError(
                "Strict data-contract failed: primary is not dominant enough. "
                f"Observed min={min_db:.2f} dB, required >= {min_primary_dominance_db:.2f} dB."
            )


# ==========================================================================
#  Training Loop
# ==========================================================================

def run_epoch(
    model: nn.Module,
    epoch: int,
    loader,
    split: str,
    opt: optim.Optimizer,
    losses: Loss,
    istft_fn: nn.Module,
    device: torch.device,
    config: Config,
    lr_scheduler_values: Optional[np.ndarray] = None,
    wd_scheduler_values: Optional[np.ndarray] = None,
    max_steps: Optional[int] = None,
    step_callback: Optional[Callable[[str, int, Dict[str, float]], None]] = None,
    val_metric_tracker: Optional[ValidationMetricTracker] = None,
) -> float:
    global debug

    log_freq = config.train.log_freq
    is_train = split == "train"
    model.train(mode=is_train)
    losses.store_losses = debug or not is_train
    max_steps_epoch = loader.len(split) - 1
    if max_steps is not None:
        max_steps_epoch = min(max_steps_epoch, max_steps)
    seed = epoch if is_train else 42
    n_nans = 0
    start_steps = epoch * loader.len(split)
    l_mem = []

    for i, batch in enumerate(loader.iter_epoch(split, seed)):
        if i > max_steps_epoch:
            break

        opt.zero_grad()
        it = start_steps + i

        # LR / WD scheduling (per-step)
        if lr_scheduler_values is not None or wd_scheduler_values is not None:
            for param_group in opt.param_groups:
                if lr_scheduler_values is not None and it < len(lr_scheduler_values):
                    param_group["lr"] = lr_scheduler_values[it] * param_group.get("lr_scale", 1)
                if wd_scheduler_values is not None and param_group["weight_decay"] > 0:
                    if it < len(wd_scheduler_values):
                        param_group["weight_decay"] = wd_scheduler_values[it]

        # Load batch data
        assert batch.feat_spec is not None
        assert batch.feat_erb is not None
        feat_erb = batch.feat_erb.to(device, non_blocking=True)
        feat_spec = as_real(batch.feat_spec.to(device, non_blocking=True))
        noisy = batch.noisy.to(device, non_blocking=True)
        clean = batch.speech.to(device, non_blocking=True)
        if hasattr(batch, "interferer") and batch.interferer is not None:
            interferer = batch.interferer.to(device, non_blocking=True)
        else:
            if getattr(losses, "sepl", None) is not None:
                raise RuntimeError(
                    "Batch is missing 'interferer' tensor while SeparationLoss is enabled. "
                    "Use the Advanced libdfdata build and verify dataloader contract."
                )
            interferer = torch.zeros_like(clean)
        snrs = batch.snr.to(device, non_blocking=True)

        with set_detect_anomaly(config.train.detect_anomaly and is_train), set_grad_enabled(is_train):
            if not is_train:
                input_spec = as_real(noisy).clone()
            else:
                input_spec = as_real(noisy)

            # Forward
            enh, m, lsnr, df_coefs, sep_mask = model.forward(
                spec=input_spec,
                feat_erb=feat_erb,
                feat_spec=feat_spec,
            )

            sep_target = None
            if getattr(losses, "sepl", None) is not None and sep_mask is not None:
                ml = getattr(losses, "ml", None)
                if ml is not None and hasattr(ml, "erb_mask_compr"):
                    with torch.no_grad():
                        sep_target = ml.erb_mask_compr(interferer, noisy, compressed=True).detach()

            try:
                err = losses.forward(
                    clean=clean,
                    noisy=noisy,
                    enhanced=enh,
                    mask=m,
                    lsnr=lsnr,
                    snrs=snrs,
                    sep_mask=sep_mask,
                    sep_target=sep_target,
                )
            except Exception as e:
                if "nan" in str(e).lower() or "finite" in str(e).lower():
                    print(f"[WARNING] NaN in loss: {e}. Skipping.")
                    n_nans += 1
                    if n_nans > MAX_NANS:
                        raise e
                    continue
                raise e

            if is_train:
                try:
                    err.backward()
                    clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True)
                except RuntimeError as e:
                    e_str = str(e)
                    if "nan" in e_str.lower() or "non-finite" in e_str:
                        print(f"[ERROR] NaN in gradients: {e}")
                        n_nans += 1
                        if n_nans > MAX_NANS:
                            raise e
                        continue
                    else:
                        raise e
                opt.step()
            elif val_metric_tracker is not None:
                val_metric_tracker.update(
                    clean_spec=clean,
                    noisy_spec=noisy,
                    enhanced_spec=enh,
                    istft_fn=istft_fn,
                )

            detach_hidden(model)

        l_mem.append(err.detach())

        if i % log_freq == 0:
            l_mean = torch.stack(l_mem[-100:]).mean().cpu()
            l_dict = {"loss": l_mean.item()}
            if lr_scheduler_values is not None:
                l_dict["lr"] = opt.param_groups[0]["lr"]
            if wd_scheduler_values is not None:
                l_dict["wd"] = opt.param_groups[0]["weight_decay"]
            step = str(i).rjust(len(str(max_steps_epoch)))
            _log(f"[{epoch}] [{step}/{max_steps_epoch}]", l_dict)
            if step_callback is not None:
                step_callback(split, it, {k: float(v) for k, v in l_dict.items()})

    if len(l_mem) == 0:
        return 0.0
    return torch.stack(l_mem).mean().cpu().item()


def _log(prefix: str, metrics: dict):
    parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    print(f"{prefix} {', '.join(parts)}")


# ==========================================================================
#  Setup Functions
# ==========================================================================

def setup_model(config: Config, device: torch.device):
    """Create DfNet model from config."""
    p = ModelConfig(
        sr=config.df.sr,
        fft_size=config.df.fft_size,
        hop_size=config.df.hop_size,
        nb_erb=config.df.nb_erb,
        nb_df=config.df.nb_df,
        norm_tau=config.df.norm_tau,
        lsnr_max=config.df.lsnr_max,
        lsnr_min=config.df.lsnr_min,
        min_nb_freqs=config.df.min_nb_freqs,
        df_order=config.df.df_order,
        df_lookahead=config.df.df_lookahead,
        conv_lookahead=config.model.conv_lookahead,
        conv_ch=config.model.conv_ch,
        conv_depthwise=config.model.conv_depthwise,
        convt_depthwise=config.model.convt_depthwise,
        conv_kernel=(int(config.model.conv_kernel[0]), int(config.model.conv_kernel[1])),
        convt_kernel=(int(config.model.convt_kernel[0]), int(config.model.convt_kernel[1])),
        conv_kernel_inp=(int(config.model.conv_kernel_inp[0]), int(config.model.conv_kernel_inp[1])),
        emb_hidden_dim=config.model.emb_hidden_dim,
        emb_num_layers=config.model.emb_num_layers,
        emb_gru_skip_enc=config.model.emb_gru_skip_enc,
        emb_gru_skip=config.model.emb_gru_skip,
        df_hidden_dim=config.model.df_hidden_dim,
        df_gru_skip=config.model.df_gru_skip,
        df_pathway_kernel_size_t=config.model.df_pathway_kernel_size_t,
        enc_concat=config.model.enc_concat,
        df_num_layers=config.model.df_num_layers,
        df_n_iter=config.model.df_n_iter,
        lin_groups=config.model.lin_groups,
        enc_lin_groups=config.model.enc_lin_groups,
        mask_pf=config.model.mask_pf,
        pf_beta=config.model.pf_beta,
        lsnr_dropout=config.model.lsnr_dropout,
    )
    erb_widths = compute_erb_widths(p.sr, p.fft_size, p.nb_erb, p.min_nb_freqs)
    erb = erb_fb(erb_widths, p.sr, inverse=False).to(device)
    erb_inv = erb_fb(erb_widths, p.sr, inverse=True).to(device)
    model = DfNet(erb, erb_inv, config=p).to(device)
    return model, erb_widths


def setup_losses(config: Config, erb_widths, device: torch.device) -> Loss:
    """Create loss function matching DF3 configuration."""
    p = get_config()
    lc = config.loss
    dc = config.df

    # Istft for time-domain losses
    window = torch.hann_window(dc.fft_size)
    istft = Istft(dc.fft_size, dc.hop_size, window).to(device)

    loss = Loss(
        sr=dc.sr,
        fft_size=dc.fft_size,
        hop_size=dc.hop_size,
        nb_erb=dc.nb_erb,
        nb_df=dc.nb_df,
        erb_widths=erb_widths,
        istft=istft,
        lsnr_min=dc.lsnr_min,
        lsnr_max=dc.lsnr_max,
        # MaskLoss
        ml_f=lc.ml_factor,
        ml_mask=lc.ml_mask,
        ml_gamma=lc.ml_gamma,
        ml_gamma_pred=lc.ml_gamma_pred,
        ml_f_under=lc.ml_f_under,
        ml_max_freq=lc.ml_max_freq,
        # SpectralLoss
        sl_fm=lc.sl_factor_magnitude,
        sl_fc=lc.sl_factor_complex,
        sl_fu=lc.sl_factor_under,
        sl_gamma=lc.sl_gamma,
        # MultiResSpecLoss
        mrsl_f=lc.mrsl_factor,
        mrsl_fc=lc.mrsl_factor_complex,
        mrsl_gamma=lc.mrsl_gamma,
        mrsl_ffts=lc.mrsl_fft_sizes,
        # SdrLoss
        sdrl_f=lc.sdrl_factor,
        sdrl_segmental_ws=lc.sdrl_segmental_ws if lc.sdrl_segmental_ws else None,
        # LocalSnrLoss
        lsnr_f=lc.lsnr_factor,
        # SeparationLoss
        sep_f=lc.sep_factor,
    ).to(device)
    return loss


def setup_optimizer(config: Config, model: nn.Module, cp_dir: str, resume: bool) -> optim.Optimizer:
    """Create optimizer matching DF3 defaults."""
    oc = config.optim
    params = model.parameters()

    supported = {
        "adam": lambda p: optim.Adam(p, lr=oc.lr, weight_decay=oc.weight_decay,
                                     betas=oc.betas, amsgrad=oc.amsgrad),
        "adamw": lambda p: optim.AdamW(p, lr=oc.lr, weight_decay=oc.weight_decay,
                                        betas=oc.betas, amsgrad=oc.amsgrad),
        "sgd": lambda p: optim.SGD(p, lr=oc.lr, weight_decay=oc.weight_decay),
        "rmsprop": lambda p: optim.RMSprop(p, lr=oc.lr, weight_decay=oc.weight_decay),
    }
    opt = supported[oc.optimizer.lower()](params)
    print(f"Optimizer: {oc.optimizer}, lr={oc.lr}, wd={oc.weight_decay}, amsgrad={oc.amsgrad}")

    if resume and cp_dir:
        load_checkpoint(opt, "opt", cp_dir)

    for group in opt.param_groups:
        group.setdefault("initial_lr", oc.lr)
    return opt


def setup_lr_schedule(config: Config, steps_per_epoch: int) -> np.ndarray:
    oc = config.optim
    tc = config.train
    return cosine_scheduler(
        oc.lr, oc.lr_min,
        epochs=tc.max_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=oc.warmup_epochs,
        start_warmup_value=oc.lr_warmup,
        initial_ep_per_cycle=oc.lr_cycle_epochs,
        cycle_decay=oc.lr_cycle_decay,
        cycle_mul=oc.lr_cycle_mul,
    )


def setup_wd_schedule(config: Config, steps_per_epoch: int) -> Optional[np.ndarray]:
    oc = config.optim
    tc = config.train
    if oc.weight_decay_end == -1:
        return None
    return cosine_scheduler(
        oc.weight_decay, oc.weight_decay_end,
        niter_per_ep=steps_per_epoch,
        epochs=tc.max_epochs,
    )


def setup_dataloader(config: Config):
    """Create libdfdata DataLoader."""
    if not HAS_LIBDF:
        raise RuntimeError("libdfdata/libdf not available. Install DeepFilterNet first.")

    dc = config.df
    tc = config.train
    norm_alpha = get_norm_alpha(dc.sr, dc.hop_size, dc.norm_tau)

    return DataLoader


# ==========================================================================
#  Main
# ==========================================================================

def main():
    global should_stop, debug

    parser = argparse.ArgumentParser(description="Train DeepFilterNet3-SE")
    parser.add_argument("--data-config", type=str, required=True,
                        help="Path to dataset config file (dataset.cfg)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to HDF5 data directory")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base directory for logs/checkpoints")
    parser.add_argument("--config-ini", type=str, default=None,
                        help="Optional INI config file to seed dataclass config values")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a model checkpoint to initialize weights from (starts training from epoch 0 with fresh optimizer)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per epoch (for debugging)")
    parser.add_argument("--p-interfer-sp", type=float, default=None,
                        help="Background speaker interference probability")
    parser.add_argument("--p-fill-speech", type=float, default=None,
                        help="Probability to append extra speech segments to primary target")
    parser.add_argument("--sep-factor", type=float, default=None,
                        help="Auxiliary separation loss factor")
    parser.add_argument("--strict-data-contract", dest="strict_data_contract", action="store_true",
                        help="Run strict preflight checks on train batches before epoch 0 (default: enabled)")
    parser.add_argument("--no-strict-data-contract", dest="strict_data_contract", action="store_false",
                        help="Disable strict preflight checks")
    parser.add_argument("--contract-check-batches", type=int, default=24,
                        help="Number of training batches to sample for strict preflight checks")
    parser.add_argument("--min-primary-dominance-db", type=float, default=3.0,
                        help="Minimum required primary-over-interferer dominance in dB for strict checks")
    parser.add_argument("--mlflow", action="store_true", default=None,
                        help="Enable MLflow tracking")
    parser.add_argument("--no-mlflow", action="store_false", dest="mlflow",
                        help="Disable MLflow tracking")
    parser.add_argument("--mlflow-uri", type=str, default=None,
                        help="MLflow tracking URI (default: local file backend in base-dir)")
    parser.add_argument("--mlflow-experiment", type=str, default=None,
                        help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", type=str, default=None,
                        help="MLflow run name")
    parser.add_argument("--metric-batches", type=int, default=8,
                        help="Max validation batches per epoch used for objective metric evaluation")
    parser.add_argument("--metric-samples", type=int, default=64,
                        help="Max validation utterances per epoch used for objective metric evaluation")
    parser.add_argument("--metric-interval", type=int, default=1,
                        help="Compute objective metrics every N validation epochs (1 = every epoch)")
    parser.add_argument("--no-stoi", action="store_false", dest="enable_stoi",
                        help="Disable STOI/ESTOI validation metrics")
    parser.add_argument("--no-pesq", action="store_false", dest="enable_pesq",
                        help="Disable PESQ validation metric")
    parser.set_defaults(strict_data_contract=True)
    parser.set_defaults(enable_stoi=True, enable_pesq=True)
    args = parser.parse_args()

    if args.metric_interval < 1:
        raise ValueError("--metric-interval must be >= 1")
    if args.metric_batches < 0:
        raise ValueError("--metric-batches must be >= 0")
    if args.metric_samples < 0:
        raise ValueError("--metric-samples must be >= 0")

    config = get_default_config()
    if args.config_ini is not None:
        config = load_config_from_ini(args.config_ini, base=config)
        print(f"Loaded config from INI: {args.config_ini}")

    # CLI overrides
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.optim.lr = args.lr
    if args.epochs:
        config.train.max_epochs = args.epochs
    if args.device:
        config.device = args.device
    if args.p_interfer_sp is not None:
        config.train.p_interfer_sp = args.p_interfer_sp
    if args.p_fill_speech is not None:
        setattr(config.train, "p_fill_speech", args.p_fill_speech)
    if args.sep_factor is not None:
        config.loss.sep_factor = args.sep_factor
    mlflow_cfg = getattr(config, "mlflow", None)
    if args.mlflow is not None:
        if mlflow_cfg is not None:
            setattr(mlflow_cfg, "enabled", args.mlflow)
    if args.mlflow_uri is not None and mlflow_cfg is not None:
        setattr(mlflow_cfg, "tracking_uri", args.mlflow_uri)
    if args.mlflow_experiment is not None and mlflow_cfg is not None:
        setattr(mlflow_cfg, "experiment_name", args.mlflow_experiment)
    if args.mlflow_run_name is not None and mlflow_cfg is not None:
        setattr(mlflow_cfg, "run_name", args.mlflow_run_name)
    if args.debug:
        config.train.detect_anomaly = True
        debug = True

    os.makedirs(args.base_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    saved_cfg, saved_cfg_path = _read_saved_run_config(args.base_dir, checkpoint_dir)
    if saved_cfg is not None and saved_cfg_path is not None:
        _assert_run_config_compatible(config, saved_cfg, saved_cfg_path)
        print(f"Using existing run config: {saved_cfg_path}")
    else:
        _write_run_config(config, args.base_dir, checkpoint_dir)
        print(
            "Saved run config to "
            f"{os.path.join(args.base_dir, RUN_CONFIG_NAME)} and "
            f"{os.path.join(checkpoint_dir, RUN_CONFIG_NAME)}"
        )

    tb_dir = os.path.join(args.base_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir)
    mlflow_run = _setup_mlflow(config, args, checkpoint_dir)
    if mlflow_run is not None and HAS_MLFLOW and mlflow is not None:
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    if args.enable_stoi and not HAS_STOI:
        print("Warning: pystoi not installed; STOI/ESTOI metrics will be skipped.")
    if args.enable_pesq and not HAS_PESQ:
        print("Warning: pesq not installed; PESQ metric will be skipped.")

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    set_seed(config.train.seed)

    # Model
    model, erb_widths = setup_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Resume or init from pretrained
    start_epoch = 0
    if args.init_from is not None:
        state = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        print(f"Initialized weights from {args.init_from} (starting from epoch 0)")
    elif args.resume:
        start_epoch = get_start_epoch(checkpoint_dir)
        if start_epoch > 0:
            load_checkpoint(model, "model", checkpoint_dir)
            print(f"Resuming from epoch {start_epoch}")

    # DataLoader
    if not HAS_LIBDF:
        raise RuntimeError("libdfdata required. Install DeepFilterNet.")
    assert DataLoader is not None

    dc = config.df
    tc = config.train
    norm_alpha = get_norm_alpha(dc.sr, dc.hop_size, dc.norm_tau)

    bs_eval = tc.batch_size_eval if tc.batch_size_eval > 0 else tc.batch_size

    loader = DataLoader(
        ds_dir=args.data_dir,
        ds_config=args.data_config,
        sr=dc.sr,
        batch_size=tc.batch_size,
        batch_size_eval=bs_eval,
        num_workers=tc.num_workers,
        pin_memory=device.type == "cuda",
        max_len_s=tc.max_sample_len_s,
        fft_size=dc.fft_size,
        hop_size=dc.hop_size,
        nb_erb=dc.nb_erb,
        nb_spec=dc.nb_df,
        norm_alpha=norm_alpha,
        p_reverb=tc.p_reverb,
        p_bw_ext=tc.p_bw_ext,
        p_clipping=tc.p_clipping,
        p_zeroing=tc.p_zeroing,
        p_air_absorption=tc.p_air_absorption,
        p_interfer_sp=tc.p_interfer_sp,
        p_fill_speech=getattr(tc, "p_fill_speech", 0.0),
        prefetch=tc.prefetch,
        seed=tc.seed,
        min_nb_erb_freqs=dc.min_nb_freqs,
        log_timings=False,
        global_sampling_factor=getattr(tc, "global_sampling_factor", 1.0),
        snrs=tc.snrs,
        gains=tc.gains,
        interfer_snrs=getattr(tc, "interfer_snrs", None),
        log_level="DEBUG" if debug else "INFO",
    )

    print(f"Train batches: {loader.len('train')}, Val batches: {loader.len('valid')}")

    if args.strict_data_contract:
        run_data_contract_preflight(
            loader=loader,
            seed=tc.seed,
            n_batches=args.contract_check_batches,
            p_interfer_sp=tc.p_interfer_sp,
            min_primary_dominance_db=args.min_primary_dominance_db,
        )

    # Optimizer
    opt = setup_optimizer(config, model, checkpoint_dir, args.resume)

    # LR / WD scheduling
    lrs = setup_lr_schedule(config, loader.len("train"))
    wds = setup_wd_schedule(config, loader.len("train"))

    # Loss
    losses = setup_losses(config, erb_widths, device)

    # Validation criteria
    patience = tc.early_stopping_patience
    val_criteria_rule = tc.validation_criteria_rule.replace("less", "min").replace("more", "max")

    # Istft for summaries
    window = torch.hann_window(dc.fft_size).to(device)
    istft_fn = Istft(dc.fft_size, dc.hop_size, window).to(device)

    mlflow_step_cb = None
    if mlflow_run is not None:
        def _mlflow_step_cb(split: str, step: int, metrics: Dict[str, float]) -> None:
            _safe_mlflow_log_metrics(f"{split}_step", metrics, step)

        mlflow_step_cb = _mlflow_step_cb

    run_status = "FINISHED"
    try:
        # Training loop
        for epoch in range(start_epoch, tc.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{tc.max_epochs}")
            print(f"{'='*60}")

            train_loss = run_epoch(
                model=model, epoch=epoch, loader=loader, split="train",
                opt=opt, losses=losses, istft_fn=istft_fn, device=device,
                config=config, lr_scheduler_values=lrs, wd_scheduler_values=wds,
                max_steps=args.max_steps, step_callback=mlflow_step_cb,
            )
            metrics = {"loss": train_loss}
            try:
                metrics["lr"] = opt.param_groups[0]["lr"]
            except (AttributeError, IndexError):
                pass
            _log(f"[{epoch}] [train]", metrics)
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    tb_writer.add_scalar(f"train/{k}", float(v), epoch)
            _safe_mlflow_log_metrics("train_epoch", metrics, epoch)

            save_checkpoint(model, "model", checkpoint_dir, epoch + 1)
            save_checkpoint(opt, "opt", checkpoint_dir, epoch + 1)
            losses.reset_summaries()

            # Validation
            metric_tracker = None
            if epoch % args.metric_interval == 0:
                metric_tracker = ValidationMetricTracker(
                    sr=dc.sr,
                    max_batches=args.metric_batches,
                    max_samples=args.metric_samples,
                    enable_stoi=args.enable_stoi,
                    enable_pesq=args.enable_pesq,
                )
            val_loss = run_epoch(
                model=model, epoch=epoch, loader=loader, split="valid",
                opt=opt, losses=losses, istft_fn=istft_fn, device=device,
                config=config, max_steps=args.max_steps, step_callback=mlflow_step_cb,
                val_metric_tracker=metric_tracker,
            )
            val_metrics = {"loss": val_loss}
            val_metrics.update(
                {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
            )
            if metric_tracker is not None:
                val_metrics.update(metric_tracker.summary())
            val_criteria = val_metrics[tc.validation_criteria]
            save_checkpoint(
                model, "model", checkpoint_dir, epoch + 1,
                metric=val_criteria, cmp=val_criteria_rule
            )
            _log(f"[{epoch}] [valid]", val_metrics)
            for k, v in val_metrics.items():
                if isinstance(v, (float, int)):
                    tb_writer.add_scalar(f"valid/{k}", float(v), epoch)
            _safe_mlflow_log_metrics("valid_epoch", val_metrics, epoch)
            tb_writer.flush()

            if not check_patience(
                checkpoint_dir, max_patience=patience,
                new_metric=val_criteria, cmp=val_criteria_rule
            ):
                print(f"Early stopping after {patience} epochs without improvement")
                break

            if should_stop:
                print("Stopping due to signal")
                break

            losses.reset_summaries()

        print("Training complete!")
    except Exception:
        run_status = "FAILED"
        raise
    finally:
        _finalize_mlflow(config, checkpoint_dir, status=run_status)
        tb_writer.close()


def get_sigusr1_handler(base_dir):
    def h(*__args):
        global should_stop
        print("Received timeout signal. Stopping after current epoch.")
        should_stop = True
    return h


if __name__ == "__main__":
    main()
