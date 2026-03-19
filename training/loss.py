"""
Loss functions for DeepFilterNet3-SE training.

Ported from the original df/loss.py with the following changes:
  1. Self-contained: no dependency on df.config/df.model/libdf for loss classes.
     ERB widths and model params are passed directly.
  2. SeparationLoss added for the auxiliary speaker separation head.
  3. Removed ASRLoss and STOI summaries (not used in standard DF3 training).

Everything else (MaskLoss, SpectralLoss, MultiResSpecLoss, SdrLoss,
SegSdrLoss, DfAlphaLoss, LocalSnrLoss, Loss wrapper) matches DF3.
"""

import warnings
from collections import defaultdict
from typing import Dict, Final, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def as_complex(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dim must be 2 for real→complex, got {x.shape[-1]}")
    return torch.view_as_complex(x.contiguous())


def as_real(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


class AngleFn(torch.autograd.Function):
    """Angle with safe gradient (avoids NaN for zero-magnitude complex)."""
    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(
            torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1)
        )

angle = AngleFn


# ---------------------------------------------------------------------------
#  ERB filterbank (rectangular, matching df/modules.py)
# ---------------------------------------------------------------------------
def make_erb_fb(erb_widths, sr: int, inverse: bool = False, normalized: bool = True) -> Tensor:
    """Create rectangular ERB filterbank from precomputed widths."""
    import numpy as np
    n_freqs = int(np.sum(erb_widths))
    all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]
    b_pts = np.cumsum([0] + list(erb_widths)).astype(int)[:-1]
    fb = torch.zeros((all_freqs.shape[0], len(b_pts)))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), list(erb_widths))):
        fb[b: b + w, i] = 1
    if inverse:
        fb = fb.t()
        if not normalized:
            fb /= fb.sum(dim=1, keepdim=True)
    else:
        if normalized:
            fb /= fb.sum(dim=0)
    return fb


# ---------------------------------------------------------------------------
#  Mask target functions (exact copies from df/loss.py)
# ---------------------------------------------------------------------------

def wg(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS = as_complex(S).abs().square()
    NN = as_complex(N).abs().square()
    return (SS / (SS + NN + eps)).clamp(0, 1)


def irm(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS_mag = as_complex(S).abs()
    NN_mag = as_complex(N).abs()
    return (SS_mag / (SS_mag + NN_mag + eps)).clamp(0, 1)


def iam(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    SS_mag = as_complex(S).abs()
    XX_mag = as_complex(X).abs()
    return (SS_mag / (XX_mag + eps)).clamp(0, 1)


# ---------------------------------------------------------------------------
#  STFT / ISTFT helpers for multi-resolution spectral loss
# ---------------------------------------------------------------------------

class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(
            input.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            normalized=True,
            return_complex=True,
        )
        out = out.view(*sh, *out.shape[-2:])
        return out


class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Tensor):
        super().__init__()
        self.n_fft_inv = n_fft_inv
        self.hop_inv = hop_inv
        self.w_inv: Tensor
        assert window_inv.shape[0] == n_fft_inv
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        input = as_complex(input)
        t, f = input.shape[-2:]
        sh = input.shape[:-2]
        out = torch.istft(
            F.pad(input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft_inv,
            hop_length=self.hop_inv,
            window=self.w_inv,
            normalized=True,
        )
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out


# ---------------------------------------------------------------------------
#  Multi-Resolution Spectral Loss
# ---------------------------------------------------------------------------

class MultiResSpecLoss(nn.Module):
    gamma: Final[float]
    f: Final[float]
    f_complex: Final[Optional[List[float]]]

    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: Optional[Union[float, Iterable[float]]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * self.f_complex[i]
        return loss


# ---------------------------------------------------------------------------
#  Spectral Loss
# ---------------------------------------------------------------------------

class SpectralLoss(nn.Module):
    gamma: Final[float]
    f_m: Final[float]
    f_c: Final[float]
    f_u: Final[float]

    def __init__(
        self,
        gamma: float = 1,
        factor_magnitude: float = 1,
        factor_complex: float = 1,
        factor_under: float = 1,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, input, target):
        input = as_complex(input)
        target = as_complex(target)
        input_abs = input.abs()
        target_abs = target.abs()
        if self.gamma != 1:
            input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
            target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)
        tmp = (input_abs - target_abs).pow(2)
        if self.f_u != 1:
            tmp *= torch.where(input_abs < target_abs, self.f_u, 1.0)
        loss = torch.mean(tmp) * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                input = input_abs * torch.exp(1j * angle.apply(input))
                target = target_abs * torch.exp(1j * angle.apply(target))
            loss_c = (
                F.mse_loss(torch.view_as_real(input), target=torch.view_as_real(target)) * self.f_c
            )
            loss = loss + loss_c
        return loss


# ---------------------------------------------------------------------------
#  Mask Loss (exact DF3 port)
# ---------------------------------------------------------------------------

class MaskLoss(nn.Module):
    def __init__(
        self,
        erb_widths,
        sr: int,
        mask: str = "iam",
        gamma: float = 0.6,
        powers: List[int] = [2],
        factors: List[float] = [1],
        f_under: float = 1,
        eps=1e-12,
        factor: float = 1.0,
        gamma_pred: Optional[float] = None,
        f_max_idx: Optional[int] = None,
    ):
        super().__init__()
        if mask == "wg":
            self.mask_fn = wg
        elif mask == "irm":
            self.mask_fn = irm
        elif mask == "iam":
            self.mask_fn = iam
        elif mask == "spec":
            self.mask_fn = None
        else:
            raise ValueError(f"Unsupported mask function: {mask}.")
        self.gamma = gamma
        self.gamma_pred = gamma if gamma_pred is None else gamma_pred
        self.powers = powers
        self.factors = factors
        self.f_under = f_under
        self.eps = eps
        self.factor = factor
        self.f_max_idx = f_max_idx
        self.erb_fb: Tensor
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_fb", make_erb_fb(erb_widths, sr))
        self.register_buffer("erb_inv_fb", make_erb_fb(erb_widths, sr, inverse=True))

    def erb_mask_compr(self, clean: Tensor, noisy: Tensor, compressed: bool = True) -> Tensor:
        mask_fn = self.mask_fn or iam
        mask = mask_fn(clean, noisy)
        mask = self.erb(mask)
        if compressed:
            mask = mask.pow(self.gamma)
        return mask

    def erb(self, x: Tensor, clamp_min: Optional[float] = None) -> Tensor:
        x = torch.matmul(x, self.erb_fb)
        if clamp_min is not None:
            x = x.clamp_min(clamp_min)
        return x

    def erb_inv(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.erb_inv_fb)

    def forward(
        self, input: Tensor, clean: Tensor, noisy: Tensor, max_bin: Optional[Tensor] = None
    ) -> Tensor:
        b, _, _, f = input.shape
        if not torch.isfinite(input).all():
            raise ValueError("Input is NaN")
        assert input.min() >= 0
        if self.mask_fn is not None:
            g_t = self.erb_mask_compr(clean, noisy, compressed=True)
            g_p = input.clamp_min(self.eps).pow(self.gamma_pred)
        else:
            g_t = self.erb(clean.abs()).pow(self.gamma)
            g_p = (self.erb(noisy.abs()) * input).pow(self.gamma_pred)
        loss = torch.zeros((), device=input.device)
        if self.f_max_idx is not None:
            g_t = g_t[..., :self.f_max_idx]
            g_p = g_p[..., :self.f_max_idx]
        tmp = g_t.sub(g_p).pow(2)
        if self.f_under != 1:
            tmp = tmp * torch.where(g_p < g_t, self.f_under, 1.0)
        if max_bin is not None:
            m = torch.ones((b, 1, 1, f), device=input.device)
            for i, mb in enumerate(max_bin):
                m[i, ..., mb:] = 0
            tmp = tmp * m
        for power, factor in zip(self.powers, self.factors):
            loss += tmp.clamp_min(1e-13).pow(power // 2).mean().mul(factor) * self.factor
        return loss.mean()


# ---------------------------------------------------------------------------
#  MaskSpecLoss
# ---------------------------------------------------------------------------

class MaskSpecLoss(nn.Module):
    def __init__(self, erb_widths, sr: int, factor=1.0, gamma: float = 0.6,
                 f_max_idx: Optional[int] = None):
        super().__init__()
        from model.dfnet_se import Mask as MaskModule
        self.f_max_idx = f_max_idx
        self.apply_mask = MaskModule(make_erb_fb(erb_widths, sr, inverse=True))
        self.loss = SpectralLoss(factor_magnitude=factor, gamma=gamma)

    def forward(self, input: Tensor, clean: Tensor, noisy: Tensor) -> Tensor:
        enh = self.apply_mask(noisy, input)
        if self.f_max_idx is not None:
            enh = enh[..., :self.f_max_idx]
            clean = clean[..., :self.f_max_idx]
        return self.loss(enh, clean)


# ---------------------------------------------------------------------------
#  DfAlphaLoss
# ---------------------------------------------------------------------------

class DfAlphaLoss(nn.Module):
    """Penalty for using DF in very noisy segments."""
    factor: Final[float]
    lsnr_thresh: Final[float]
    lsnr_min: Final[float]

    def __init__(self, factor: float = 1, lsnr_thresh: float = -7.5, lsnr_min: float = -10.0):
        super().__init__()
        self.factor = factor
        self.lsnr_thresh = lsnr_thresh
        self.lsnr_min = lsnr_min

    def forward(self, pred_alpha: Tensor, target_lsnr: Tensor):
        w = self.lsnr_mapping(target_lsnr, self.lsnr_thresh, self.lsnr_min).view_as(pred_alpha)
        l_off = (pred_alpha * w).square().mean()
        w = self.lsnr_mapping(target_lsnr, self.lsnr_thresh + 2.5, 0.0).view_as(pred_alpha)
        l_on = 0.1 * ((1 - pred_alpha) * w).abs().mean()
        return l_off + l_on

    def lsnr_mapping(self, lsnr: Tensor, lsnr_thresh: float,
                     lsnr_min: Optional[float] = None) -> Tensor:
        lsnr_min = float(self.lsnr_min) if lsnr_min is None else lsnr_min
        a_ = 1 / (lsnr_thresh - lsnr_min)
        b_ = -a_ * lsnr_min
        return 1 - torch.clamp(a_ * lsnr + b_, 0.0, 1.0)


# ---------------------------------------------------------------------------
#  SDR Losses
# ---------------------------------------------------------------------------

class SiSdr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        eps = torch.finfo(input.dtype).eps
        t = input.shape[-1]
        target = target.reshape(-1, t)
        input = input.reshape(-1, t)
        Rss: Tensor = torch.einsum("bi,bi->b", target, target).unsqueeze(-1)
        a: Tensor = torch.einsum("bi,bi->b", target, input).add(eps).unsqueeze(-1) / Rss.add(eps)
        e_true = a * target
        e_res = input - e_true
        Sss = e_true.square().sum(-1)
        Snn = e_res.square().sum(-1)
        return 10 * torch.log10(Sss.add(eps) / Snn.add(eps))


def si_sdr(input: Tensor, target: Tensor) -> Tensor:
    """Convenience SI-SDR metric function."""
    return SiSdr()(input, target).mean()


class SdrLoss(nn.Module):
    def __init__(self, factor=0.2):
        super().__init__()
        self.factor = factor
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        return -self.sdr(input, target).mean() * self.factor


class SegSdrLoss(nn.Module):
    def __init__(self, window_sizes: List[int], factor: float = 0.2, overlap: float = 0):
        super().__init__()
        self.window_sizes = window_sizes
        self.factor = factor
        self.hop = 1 - overlap
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        loss = torch.zeros((), device=input.device)
        for ws in self.window_sizes:
            if ws > input.size(-1):
                warnings.warn(
                    f"Input size {input.size(-1)} smaller than window size. Adjusting."
                )
                ws = input.size(1)
            loss += self.sdr(
                input=input.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
                target=target.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
            ).mean()
        return -loss * self.factor


# ---------------------------------------------------------------------------
#  Local SNR Loss and Target
# ---------------------------------------------------------------------------

class LocalSnrLoss(nn.Module):
    def __init__(self, factor: float = 1):
        super().__init__()
        self.factor = factor

    def forward(self, input: Tensor, target_lsnr: Tensor):
        input = input.squeeze(-1)
        return F.mse_loss(input, target_lsnr) * self.factor


def _local_energy(x: Tensor, ws: int, device: torch.device) -> Tensor:
    if (ws % 2) == 0:
        ws += 1
    ws_half = ws // 2
    x = F.pad(x.pow(2).sum(-1).sum(-1), (ws_half, ws_half, 0, 0))
    w = torch.hann_window(ws, device=device, dtype=x.dtype)
    x = x.unfold(-1, size=ws, step=1) * w
    return torch.sum(x, dim=-1).div(ws)


def local_snr(
    clean: Tensor,
    noise: Tensor,
    window_size: int,
    db: bool = False,
    window_size_ns: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    clean = as_real(clean)
    noise = as_real(noise)
    assert clean.dim() == 5
    E_speech = _local_energy(clean, window_size, clean.device)
    window_size_ns = window_size if window_size_ns is None else window_size_ns
    E_noise = _local_energy(noise, window_size_ns, clean.device)
    snr = E_speech / E_noise.clamp_min(eps)
    if db:
        snr = snr.clamp_min(eps).log10().mul(10)
    return snr, E_speech, E_noise


class LocalSnrTarget(nn.Module):
    def __init__(self, sr: int, fft_size: int, hop_size: int,
                 ws: int = 20, db: bool = True, ws_ns: Optional[int] = None,
                 target_snr_range=None):
        super().__init__()
        self.sr = sr
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.ws = self._calc_ws(ws)
        self.ws_ns = self.ws * 2 if ws_ns is None else self._calc_ws(ws_ns)
        self.db = db
        self.range = target_snr_range

    def _calc_ws(self, ws_ms: int) -> int:
        ws = ws_ms - self.fft_size / self.sr * 1000
        ws = 1 + ws / (self.hop_size / self.sr * 1000)
        return max(int(round(ws)), 1)

    def forward(self, clean: Tensor, noise: Tensor, max_bin: Optional[int] = None) -> Tensor:
        if max_bin is not None:
            clean = as_complex(clean[..., :max_bin])
            noise = as_complex(noise[..., :max_bin])
        return (
            local_snr(clean, noise, window_size=self.ws, db=self.db, window_size_ns=self.ws_ns)[0]
            .clamp(self.range[0], self.range[1])
            .squeeze(1)
        )


# ---------------------------------------------------------------------------
#  Separation Loss (intentional addition – not in DF3)
# ---------------------------------------------------------------------------

class SeparationLoss(nn.Module):
    """L1 loss on ERB-domain separation mask."""

    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor

    def forward(self, pred_mask: Tensor, target_mask: Tensor) -> Tensor:
        if self.factor == 0:
            return torch.zeros((), device=pred_mask.device)
        return F.l1_loss(pred_mask, target_mask) * self.factor


# ---------------------------------------------------------------------------
#  Combined Loss class (mirrors DF3 Loss wrapper)
# ---------------------------------------------------------------------------

class Loss(nn.Module):
    """Loss wrapper matching DF3's config-driven loss composition.

    Args:
        sr: Sample rate
        fft_size: FFT size
        hop_size: Hop size
        nb_erb: Number of ERB bands
        nb_df: Number of DF bins
        erb_widths: ERB band widths (numpy array)
        istft: Istft module for time-domain losses
        lsnr_min: Min local SNR
        lsnr_max: Max local SNR

        Loss component config (all default to DF3 values):
        ml_f: MaskLoss factor
        sl_fm: SpectralLoss magnitude factor
        sl_fc: SpectralLoss complex factor
        sl_fu: SpectralLoss under-estimation factor
        sl_gamma: SpectralLoss gamma
        mrsl_f: MultiResSpecLoss factor
        mrsl_fc: MultiResSpecLoss complex factor
        mrsl_gamma: MultiResSpecLoss gamma
        mrsl_ffts: MultiResSpecLoss FFT sizes
        sdrl_f: SdrLoss factor
        lsnr_f: LocalSnrLoss factor
        sep_f: SeparationLoss factor
    """

    def __init__(
        self,
        sr: int = 16000,
        fft_size: int = 320,
        hop_size: int = 160,
        nb_erb: int = 32,
        nb_df: int = 64,
        erb_widths=None,
        istft: Optional[Istft] = None,
        lsnr_min: int = -15,
        lsnr_max: int = 35,
        # MaskLoss config (DF3 defaults)
        ml_f: float = 1.0,
        ml_mask: str = "iam",
        ml_gamma: float = 0.6,
        ml_gamma_pred: float = 0.6,
        ml_f_under: float = 2.0,
        ml_max_freq: float = 0.0,
        # SpectralLoss config
        sl_fm: float = 1e4,
        sl_fc: float = 0.0,
        sl_fu: float = 1.0,
        sl_gamma: float = 1.0,
        # MultiResSpecLoss config
        mrsl_f: float = 0.0,
        mrsl_fc: float = 0.0,
        mrsl_gamma: float = 1.0,
        mrsl_ffts: List[int] = [512, 1024, 2048],
        # SdrLoss config
        sdrl_f: float = 0.0,
        sdrl_segmental_ws: Optional[List[int]] = None,
        # LocalSnrLoss config
        lsnr_f: float = 0.0005,
        # SeparationLoss config (intentional addition)
        sep_f: float = 0.5,
    ):
        super().__init__()

        # Compute erb_widths if not provided
        if erb_widths is None:
            from model.dfnet_se import compute_erb_widths
            erb_widths = compute_erb_widths(sr, fft_size, nb_erb)

        self.sr = sr
        self.fft_size = fft_size
        self.nb_df = nb_df
        self.istft = istft
        self.store_losses = False
        self.summaries: Dict[str, List[Tensor]] = self.reset_summaries()

        # LocalSnrTarget
        self.lsnr = LocalSnrTarget(
            sr=sr, fft_size=fft_size, hop_size=hop_size,
            ws=20, target_snr_range=[lsnr_min - 1, lsnr_max + 1]
        )

        # MaskLoss
        self.ml_f = ml_f
        ml_f_max_idx = None
        if ml_max_freq > 0:
            ml_f_max_idx = int(ml_max_freq / (sr / fft_size))

        if ml_mask == "spec":
            self.ml = MaskSpecLoss(erb_widths, sr, ml_f, ml_gamma, f_max_idx=ml_f_max_idx)
        else:
            self.ml = MaskLoss(
                erb_widths, sr,
                mask=ml_mask,
                factor=ml_f,
                f_under=ml_f_under,
                gamma=ml_gamma,
                gamma_pred=ml_gamma_pred,
                factors=[1, 10],
                powers=[2, 4],
                f_max_idx=ml_f_max_idx,
            )

        # SpectralLoss
        self.sl_f = sl_fm + sl_fc
        self.sl = None
        if self.sl_f > 0:
            self.sl = SpectralLoss(
                factor_magnitude=sl_fm,
                factor_complex=sl_fc,
                factor_under=sl_fu,
                gamma=sl_gamma,
            )

        # MultiResSpecLoss
        self.mrsl_f = mrsl_f
        self.mrsl = None
        if self.mrsl_f > 0:
            assert istft is not None
            self.mrsl = MultiResSpecLoss(mrsl_ffts, mrsl_gamma, mrsl_f, mrsl_fc)

        # SdrLoss
        self.sdrl_f = sdrl_f
        self.sdrl = None
        if self.sdrl_f > 0:
            if sdrl_segmental_ws and any(ws > 0 for ws in sdrl_segmental_ws):
                self.sdrl = SegSdrLoss(sdrl_segmental_ws, factor=self.sdrl_f)
            else:
                self.sdrl = SdrLoss(self.sdrl_f)

        # LocalSnrLoss
        self.lsnr_f = lsnr_f
        self.lsnrl = LocalSnrLoss(self.lsnr_f) if self.lsnr_f > 0 else None

        # SeparationLoss (intentional addition)
        self.sep_f = sep_f
        self.sepl = SeparationLoss(self.sep_f) if self.sep_f > 0 else None

    def forward(
        self,
        clean: Tensor,
        noisy: Tensor,
        enhanced: Tensor,
        mask: Tensor,
        lsnr: Tensor,
        snrs: Tensor,
        sep_mask: Optional[Tensor] = None,
        sep_target: Optional[Tensor] = None,
        max_freq: Optional[Tensor] = None,
    ):
        """Compute all losses.

        Args:
            clean: Clean complex spectrum [B, 1, T, F, 2]
            noisy: Noisy complex spectrum [B, 1, T, F, 2]
            enhanced: Enhanced complex spectrum [B, 1, T, F, 2]
            mask: ERB mask estimate [B, 1, T, E]
            lsnr: Local SNR estimate [B, T, 1]
            snrs: Input SNRs [B]
            sep_mask: Separation mask prediction [B, 1, T, E] (optional)
            sep_target: Separation mask target [B, 1, T, E] (optional)
        """
        enhanced_td = None
        clean_td = None
        lsnr_gt = self.lsnr(clean, noise=noisy - clean)
        if self.istft is not None:
            if self.store_losses or self.mrsl is not None or self.sdrl is not None:
                enhanced_td = self.istft(enhanced)
                clean_td = self.istft(clean)

        ml = torch.zeros((), device=clean.device)
        sl = torch.zeros((), device=clean.device)
        mrsl = torch.zeros((), device=clean.device)
        sdrl = torch.zeros((), device=clean.device)
        lsnrl = torch.zeros((), device=clean.device)
        sepl = torch.zeros((), device=clean.device)

        if self.ml_f != 0 and self.ml is not None:
            ml = self.ml(input=mask, clean=clean, noisy=noisy)
        if self.sl_f != 0 and self.sl is not None:
            sl = self.sl(input=enhanced, target=clean)
        if self.mrsl_f > 0 and self.mrsl is not None:
            mrsl = self.mrsl(enhanced_td, clean_td)
        if self.lsnr_f != 0 and self.lsnrl is not None:
            lsnrl = self.lsnrl(input=lsnr, target_lsnr=lsnr_gt)
        if self.sdrl_f != 0 and self.sdrl is not None:
            sdrl = self.sdrl(enhanced_td, clean_td)
        if self.sep_f != 0 and self.sepl is not None and sep_mask is not None and sep_target is not None:
            sepl = self.sepl(sep_mask, sep_target)

        if self.store_losses:
            self.store_summaries(ml, sl, mrsl, sdrl, lsnrl, sepl)

        return ml + sl + mrsl + sdrl + lsnrl + sepl

    def reset_summaries(self):
        self.summaries = defaultdict(list)
        return self.summaries

    def get_summaries(self):
        return self.summaries.items()

    @torch.no_grad()
    def store_summaries(self, ml, sl, mrsl, sdrl, lsnrl, sepl):
        if ml != 0:
            self.summaries["MaskLoss"].append(ml.detach())
        if sl != 0:
            self.summaries["SpectralLoss"].append(sl.detach())
        if mrsl != 0:
            self.summaries["MultiResSpecLoss"].append(mrsl.detach())
        if sdrl != 0:
            self.summaries["SdrLoss"].append(sdrl.detach())
        if lsnrl != 0:
            self.summaries["LocalSnrLoss"].append(lsnrl.detach())
        if sepl != 0:
            self.summaries["SeparationLoss"].append(sepl.detach())
