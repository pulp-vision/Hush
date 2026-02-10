#!/usr/bin/env python3
"""Regression check for model/loss/training interface compatibility."""

import sys

import torch


def main() -> int:
    repo = "/home/ubuntu/Noise-Cancellation"
    if repo not in sys.path:
        sys.path.append(repo)

    from model.dfnet_se import DfNet, DfNetSE, ModelConfig, compute_erb_widths, erb_fb
    from training.loss import Loss

    cfg = ModelConfig()
    widths = compute_erb_widths(cfg.sr, cfg.fft_size, cfg.nb_erb, cfg.min_nb_freqs)
    erb = erb_fb(widths, cfg.sr, inverse=False)
    erb_inv = erb_fb(widths, cfg.sr, inverse=True)

    # Waveform API check.
    model_wav = DfNetSE(cfg)
    wav = torch.randn(2, 1, cfg.sr)
    enh_wav = model_wav(wav)
    assert enh_wav.shape == wav.shape, f"Waveform output mismatch: {enh_wav.shape} != {wav.shape}"

    # Spec-domain training API check.
    model = DfNet(erb, erb_inv, config=cfg)
    b, t = 2, 32
    f = cfg.fft_size // 2 + 1
    clean = torch.randn(b, 1, t, f, 2)
    noisy = clean + 0.1 * torch.randn_like(clean)
    feat_erb = torch.randn(b, 1, t, cfg.nb_erb)
    feat_spec = torch.randn(b, 1, t, cfg.nb_df, 2)

    enhanced, mask, lsnr, _, sep_mask = model(noisy, feat_erb, feat_spec)
    assert enhanced.shape == clean.shape
    assert mask.shape == (b, 1, t, cfg.nb_erb)
    assert lsnr.shape == (b, t, 1)
    assert sep_mask.shape == (b, 1, t, cfg.nb_erb)

    # Loss/sep-target path check.
    loss_mod = Loss(
        sr=cfg.sr,
        fft_size=cfg.fft_size,
        hop_size=cfg.hop_size,
        nb_erb=cfg.nb_erb,
        nb_df=cfg.nb_df,
        erb_widths=widths,
        sep_f=0.5,
    )
    loss_mod.store_losses = True
    sep_target = loss_mod.ml.erb_mask_compr(clean, noisy, compressed=True)
    snrs = torch.tensor([0.0, 5.0])
    total = loss_mod(clean, noisy, enhanced, mask, lsnr, snrs, sep_mask=sep_mask, sep_target=sep_target)
    assert torch.isfinite(total).item(), "Total loss is not finite"
    assert "SeparationLoss" in dict(loss_mod.get_summaries()), "SeparationLoss summary missing"

    print("regression_interface_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
