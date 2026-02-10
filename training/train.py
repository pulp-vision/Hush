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
import math
import os
import random
import signal
import sys
import site
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    env["LD_LIBRARY_PATH"] = f"{cudnn_dir}:{cublas_dir}"
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
from training.config import Config, get_default_config
from training.loss import Istft, Loss
from training.lr import cosine_scheduler

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_norm_alpha(sr: int, hop_size: int, tau: float) -> float:
    return math.exp(-hop_size / sr / tau)


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
        obj.load_state_dict(torch.load(path, map_location="cpu"))
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
                        sep_target = ml.erb_mask_compr(clean, noisy, compressed=True).detach()

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
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per epoch (for debugging)")
    parser.add_argument("--p-interfer-sp", type=float, default=None,
                        help="Background speaker interference probability")
    parser.add_argument("--sep-factor", type=float, default=None,
                        help="Auxiliary separation loss factor")
    args = parser.parse_args()

    config = get_default_config()

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
    if args.sep_factor is not None:
        config.loss.sep_factor = args.sep_factor
    if args.debug:
        config.train.detect_anomaly = True
        debug = True

    os.makedirs(args.base_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tb_dir = os.path.join(args.base_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    set_seed(config.train.seed)

    # Model
    model, erb_widths = setup_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Resume
    start_epoch = 0
    if args.resume:
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
        prefetch=tc.prefetch,
        seed=tc.seed,
        min_nb_erb_freqs=dc.min_nb_freqs,
        log_timings=False,
        snrs=tc.snrs,
        gains=tc.gains,
        log_level="DEBUG" if debug else "INFO",
    )

    print(f"Train batches: {loader.len('train')}, Val batches: {loader.len('valid')}")

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

    # Training loop
    for epoch in range(start_epoch, tc.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{tc.max_epochs}")
        print(f"{'='*60}")

        train_loss = run_epoch(
            model=model, epoch=epoch, loader=loader, split="train",
            opt=opt, losses=losses, istft_fn=istft_fn, device=device,
            config=config, lr_scheduler_values=lrs, wd_scheduler_values=wds,
            max_steps=args.max_steps,
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

        save_checkpoint(model, "model", checkpoint_dir, epoch + 1)
        save_checkpoint(opt, "opt", checkpoint_dir, epoch + 1)
        losses.reset_summaries()

        # Validation
        val_loss = run_epoch(
            model=model, epoch=epoch, loader=loader, split="valid",
            opt=opt, losses=losses, istft_fn=istft_fn, device=device,
            config=config, max_steps=args.max_steps,
        )
        val_metrics = {"loss": val_loss}
        val_metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        val_criteria = val_metrics[tc.validation_criteria]
        save_checkpoint(
            model, "model", checkpoint_dir, epoch + 1,
            metric=val_criteria, cmp=val_criteria_rule
        )
        _log(f"[{epoch}] [valid]", val_metrics)
        for k, v in val_metrics.items():
            if isinstance(v, (float, int)):
                tb_writer.add_scalar(f"valid/{k}", float(v), epoch)
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
    tb_writer.close()


def get_sigusr1_handler(base_dir):
    def h(*__args):
        global should_stop
        print("Received timeout signal. Stopping after current epoch.")
        should_stop = True
    return h


if __name__ == "__main__":
    main()
