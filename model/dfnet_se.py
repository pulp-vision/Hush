"""
DeepFilterNet3-style model adapted for 16 kHz with an auxiliary separation head.

This file keeps DF3 core architecture and training interfaces while intentionally
adding:
1) 16 kHz defaults,
2) auxiliary ERB-domain separation head,
3) compatibility with background-speaker-interference data.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Final, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


def as_real(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


def as_complex(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return x
    return torch.view_as_complex(x.contiguous())


def get_norm_alpha(sr: int, hop_size: int, tau: float = 1.0) -> float:
    return math.exp(-hop_size / sr / tau)


def compute_erb_widths(
    sr: int,
    fft_size: int,
    nb_erb: int,
    min_nb_freqs: int = 2,
) -> np.ndarray:
    """Compute ERB widths; prefer libdf widths when available."""
    n_freqs = fft_size // 2 + 1

    try:
        from libdf import DF

        df_state = DF(
            sr=sr,
            fft_size=fft_size,
            hop_size=fft_size // 4,
            nb_bands=nb_erb,
            min_nb_erb_freqs=min_nb_freqs,
        )
        widths = np.asarray(df_state.erb_widths(), dtype=np.int64)
        if widths.sum() == n_freqs:
            return widths
    except Exception:
        pass

    # Fallback: ERB-spaced boundaries in Hz, then projected to FFT bins.
    def hz_to_erb(f_hz: np.ndarray) -> np.ndarray:
        return 21.4 * np.log10(1.0 + f_hz / 229.0)

    def erb_to_hz(erb: np.ndarray) -> np.ndarray:
        return 229.0 * (10 ** (erb / 21.4) - 1.0)

    bin_hz = np.linspace(0.0, sr / 2.0, n_freqs)
    erb_edges = np.linspace(hz_to_erb(np.array([0.0]))[0], hz_to_erb(np.array([sr / 2.0]))[0], nb_erb + 1)
    hz_edges = erb_to_hz(erb_edges)
    bin_edges = np.searchsorted(bin_hz, hz_edges, side="left")
    bin_edges[0] = 0
    bin_edges[-1] = n_freqs

    widths = np.diff(bin_edges).astype(np.int64)
    widths = np.maximum(widths, min_nb_freqs)

    # Rebalance to exact n_freqs while respecting min width.
    diff = int(n_freqs - widths.sum())
    if diff > 0:
        widths[-1] += diff
    elif diff < 0:
        overflow = -diff
        for i in range(len(widths) - 1, -1, -1):
            room = int(widths[i] - min_nb_freqs)
            if room <= 0:
                continue
            take = min(room, overflow)
            widths[i] -= take
            overflow -= take
            if overflow == 0:
                break
        if overflow > 0:
            widths[-1] = max(min_nb_freqs, widths[-1] - overflow)

    # Final one-bin correction if needed.
    widths[-1] += int(n_freqs - widths.sum())
    return widths.astype(np.int64)


def erb_fb(
    widths: np.ndarray,
    sr: int,
    normalized: bool = True,
    inverse: bool = False,
) -> Tensor:
    """DF3 rectangular ERB filterbank."""
    n_freqs = int(np.sum(widths))
    all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]
    b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]
    fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
        fb[b : b + w, i] = 1
    if inverse:
        fb = fb.t()
        if not normalized:
            fb /= fb.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        if normalized:
            fb /= fb.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return fb


@dataclass
class ModelConfig:
    # DF params (16 kHz adaptation)
    sr: int = 16000
    fft_size: int = 320
    hop_size: int = 160
    nb_erb: int = 32
    nb_df: int = 64
    norm_tau: float = 1.0
    lsnr_max: int = 35
    lsnr_min: int = -15
    min_nb_freqs: int = 2
    df_order: int = 5
    df_lookahead: int = 0

    # Model params (DF3 defaults)
    conv_lookahead: int = 0
    conv_ch: int = 16
    conv_depthwise: bool = True
    convt_depthwise: bool = True
    conv_kernel: Tuple[int, int] = (1, 3)
    convt_kernel: Tuple[int, int] = (1, 3)
    conv_kernel_inp: Tuple[int, int] = (3, 3)
    emb_hidden_dim: int = 256
    emb_num_layers: int = 2
    emb_gru_skip_enc: str = "none"
    emb_gru_skip: str = "none"
    df_hidden_dim: int = 256
    df_gru_skip: str = "none"
    df_pathway_kernel_size_t: int = 1
    enc_concat: bool = False
    df_num_layers: int = 3
    df_n_iter: int = 1
    lin_groups: int = 1
    enc_lin_groups: int = 16
    mask_pf: bool = False
    pf_beta: float = 0.02
    lsnr_dropout: bool = False


def get_config() -> ModelConfig:
    return ModelConfig()


class Conv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ):
        lookahead = 0
        if isinstance(kernel_size, int):
            k_tup: Tuple[int, int] = (kernel_size, kernel_size)
        else:
            k_tup = (int(kernel_size[0]), int(kernel_size[1]))
        fpad_ = k_tup[1] // 2 + dilation - 1 if fpad else 0
        pad = (0, 0, k_tup[0] - 1 - lookahead, lookahead)
        layers: List[nn.Module] = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1 or max(k_tup) == 1:
            separable = False
            groups = 1
        layers.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k_tup,
                padding=(0, fpad_),
                stride=(1, fstride),
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ):
        lookahead = 0
        if isinstance(kernel_size, int):
            k_tup: Tuple[int, int] = (kernel_size, kernel_size)
        else:
            k_tup = (int(kernel_size[0]), int(kernel_size[1]))
        fpad_ = k_tup[1] // 2 if fpad else 0
        pad = (0, 0, k_tup[0] - 1 - lookahead, lookahead)
        layers: List[nn.Module] = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=k_tup,
                padding=(k_tup[0] - 1, fpad_ + dilation - 1),
                output_padding=(0, fpad_),
                stride=(1, fstride),
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class GroupedLinearEinsum(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.ws = input_size // groups
        self.register_parameter(
            "weight",
            Parameter(torch.zeros(groups, input_size // groups, hidden_size // groups), requires_grad=True),
        )
        self.reset_parameters()

    def reset_parameters(self):
        weight = self.weight
        assert isinstance(weight, Tensor)
        init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        x = x.view((b, t, self.groups, self.ws))
        x = torch.einsum("btgi,gih->btgh", x, self.weight)
        return x.flatten(2, 3)


class SqueezedGRU_S(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., nn.Module]] = None,
        linear_act_layer: Callable[..., nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups),
            linear_act_layer(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups),
                linear_act_layer(),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, h = self.gru(x, h)
        assert h is not None
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h


class Mask(nn.Module):
    def __init__(self, erb_inv_fb: Tensor):
        super().__init__()
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_inv_fb", erb_inv_fb)

    def forward(self, spec: Tensor, mask: Tensor) -> Tensor:
        mask = mask.matmul(self.erb_inv_fb)
        return spec * mask.unsqueeze(-1)


class ExponentialUnitNorm(nn.Module):
    """Exponential unit norm for complex ([...,2]) and real feature tensors."""

    alpha: Final[float]
    eps: Final[float]

    def __init__(self, alpha: float, num_freq_bins: int, eps: float = 1e-14):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.init_state: Tensor
        self.register_buffer("init_state", torch.ones(1, 1, num_freq_bins, 1))

    def _forward_complex(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F, 2]
        b, c, t, f, _ = x.shape
        x_abs = x.square().sum(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        state = self.init_state[:, :, :f].clone().expand(b, c, f, 1)
        out_states: List[Tensor] = []
        for i in range(t):
            state = x_abs[:, :, i] * (1 - self.alpha) + state * self.alpha
            out_states.append(state)
        return x / torch.stack(out_states, 2).sqrt()

    def _forward_real(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F]
        b, c, t, f = x.shape
        x_abs = x.abs().unsqueeze(-1).clamp_min(self.eps)
        state = self.init_state[:, :, :f].clone().expand(b, c, f, 1)
        out_states: List[Tensor] = []
        for i in range(t):
            state = x_abs[:, :, i] * (1 - self.alpha) + state * self.alpha
            out_states.append(state)
        denom = torch.stack(out_states, 2).sqrt().squeeze(-1)
        return x / denom

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5 and x.shape[-1] == 2:
            return self._forward_complex(x)
        if x.ndim == 4:
            return self._forward_real(x)
        raise ValueError(f"Unsupported shape for ExponentialUnitNorm: {tuple(x.shape)}")


class Add(nn.Module):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b


class Concat(nn.Module):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.cat((a, b), dim=-1)


def _resolve_skip_op(kind: str, in_dim: int, out_dim: int, groups: int) -> Optional[Callable[..., nn.Module]]:
    k = kind.lower()
    if k == "none":
        return None
    if k == "identity":
        assert in_dim == out_dim, "Dimensions do not match for identity skip"
        return partial(nn.Identity)
    if k == "groupedlinear":
        return partial(GroupedLinearEinsum, input_size=in_dim, hidden_size=out_dim, groups=groups)
    raise NotImplementedError(f"Unsupported skip op: {kind}")


class Encoder(nn.Module):
    def __init__(self, p: Optional[ModelConfig] = None):
        super().__init__()
        p = p or get_config()
        assert p.nb_erb % 4 == 0

        self.erb_conv0 = Conv2dNormAct(1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True)
        conv_layer = partial(
            Conv2dNormAct,
            in_ch=p.conv_ch,
            out_ch=p.conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(2, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True)
        self.df_conv1 = conv_layer(fstride=2)

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        df_fc_emb = GroupedLinearEinsum(p.conv_ch * p.nb_df // 2, self.emb_in_dim, groups=p.enc_lin_groups)
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))

        if p.enc_concat:
            self.emb_in_dim *= 2
            self.combine: nn.Module = Concat()
        else:
            self.combine = Add()

        skip_op = _resolve_skip_op(p.emb_gru_skip_enc, self.emb_out_dim, self.emb_out_dim, p.lin_groups)
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            batch_first=True,
            gru_skip_op=skip_op,
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(self, feat_erb: Tensor, feat_spec: Tensor):
        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        c0 = self.df_conv0(feat_spec)
        c1 = self.df_conv1(c0)

        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = self.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = self.combine(emb, cemb)
        emb, _ = self.emb_gru(emb)
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self, p: Optional[ModelConfig] = None):
        super().__init__()
        p = p or get_config()
        assert p.nb_erb % 8 == 0

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        skip_op = _resolve_skip_op(p.emb_gru_skip, self.emb_in_dim, self.emb_out_dim, p.lin_groups)
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=p.emb_num_layers - 1,
            batch_first=True,
            gru_skip_op=skip_op,
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        tconv_layer = partial(ConvTranspose2dNormAct, kernel_size=p.convt_kernel, bias=False, separable=True)
        conv_layer = partial(Conv2dNormAct, bias=False, separable=True)

        self.conv3p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid)

    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor) -> Tensor:
        b, _, t, f8 = e3.shape
        emb, _ = self.emb_gru(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3 = self.convt3(self.conv3p(e3) + emb)
        e2 = self.convt2(self.conv2p(e2) + e3)
        e1 = self.convt1(self.conv1p(e1) + e2)
        m = self.conv0_out(self.conv0p(e0) + e1)
        return m


class DfOutputReshapeMF(nn.Module):
    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        shape = list(coefs.shape)
        shape[-1] = -1
        shape.append(2)
        coefs = coefs.view(shape)
        return coefs.permute(0, 3, 1, 2, 4)


class DfDecoder(nn.Module):
    def __init__(self, p: Optional[ModelConfig] = None):
        super().__init__()
        p = p or get_config()
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim
        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2

        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = p.df_pathway_kernel_size_t
        self.df_convp = conv_layer(p.conv_ch, self.df_out_ch, fstride=1, kernel_size=(kt, 1))
        self.df_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            num_layers=self.df_n_layers,
            batch_first=True,
            gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )

        skip_kind = p.df_gru_skip.lower()
        if skip_kind == "none":
            self.df_skip: Optional[nn.Module] = None
        elif skip_kind == "identity":
            assert p.emb_hidden_dim == p.df_hidden_dim
            self.df_skip = nn.Identity()
        elif skip_kind == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(self.emb_in_dim, self.emb_dim, groups=p.lin_groups)
        else:
            raise NotImplementedError()

        out_dim = self.df_bins * self.df_out_ch
        self.df_out = nn.Sequential(GroupedLinearEinsum(self.df_n_hidden, out_dim, groups=p.lin_groups), nn.Tanh())
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())

    def forward(self, emb: Tensor, c0: Tensor) -> Tensor:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)
        c = self.df_out(c)
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0
        return c


class SeparationHead(nn.Module):
    def __init__(self, emb_dim: int, nb_erb: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(emb_dim, nb_erb), nn.Sigmoid())

    def forward(self, emb: Tensor) -> Tensor:
        # [B, T, D] -> [B, 1, T, E]
        return self.fc(emb).unsqueeze(1)


def spec_pad(x: Tensor, window_size: int, lookahead: int, dim: int = 0) -> Tensor:
    pad = [0] * (x.dim() * 2)
    if dim >= 0:
        idx = (x.dim() - dim - 1) * 2
    else:
        idx = (-dim - 1) * 2
    pad[idx] = window_size - lookahead - 1
    pad[idx + 1] = lookahead
    return F.pad(x, pad)


class DfNet(nn.Module):
    run_df: Final[bool]
    run_erb: Final[bool]
    post_filter: Final[bool]

    def __init__(
        self,
        erb_fb: Tensor,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        p = config or get_config()
        self.cfg = p
        self.df_lookahead = p.df_lookahead
        self.nb_df = p.nb_df
        self.freq_bins = p.fft_size // 2 + 1
        self.erb_bins = p.nb_erb
        self.df_order = p.df_order
        self.train_mask = train_mask

        if p.conv_lookahead > 0:
            assert p.conv_lookahead >= p.df_lookahead
            self.pad_feat: nn.Module = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()

        if p.df_lookahead > 0:
            self.pad_spec: nn.Module = nn.ConstantPad3d((0, 0, 0, 0, -p.df_lookahead, p.df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()

        self.erb_fb: Tensor
        self.register_buffer("erb_fb", erb_fb)
        self.enc = Encoder(p)
        self.erb_dec = ErbDecoder(p)
        self.mask = Mask(erb_inv_fb)
        self.erb_inv_fb = erb_inv_fb
        self.post_filter = p.mask_pf
        self.post_filter_beta = p.pf_beta
        self.df_dec = DfDecoder(p)
        self.df_out_transform = DfOutputReshapeMF(self.df_order, p.nb_df)
        self.sep_head = SeparationHead(self.enc.emb_out_dim, p.nb_erb)

        self.run_erb = p.nb_df + 1 < self.freq_bins
        self.run_df = run_df
        self.lsnr_dropout = p.lsnr_dropout

    def apply_df(self, spec: Tensor, coefs: Tensor) -> Tensor:
        # spec: [B, 1, T, F, 2], coefs: [B, O, T, F_df, 2]
        spec_df = spec[..., : self.nb_df, :].squeeze(1)
        padded = spec_pad(spec_df, self.df_order, self.df_lookahead, dim=-3)
        padded = padded.unfold(dimension=1, size=self.df_order, step=1)  # [B, T, F, 2, O]
        padded = padded.permute(0, 1, 4, 2, 3)  # [B, T, O, F, 2]
        coefs = coefs.permute(0, 2, 1, 3, 4)  # [B, T, O, F, 2]

        re = padded[..., 0] * coefs[..., 0] - padded[..., 1] * coefs[..., 1]
        im = padded[..., 1] * coefs[..., 0] + padded[..., 0] * coefs[..., 1]
        spec_f = torch.stack((re, im), dim=-1).sum(dim=2)

        out = spec.clone()
        out[..., : self.nb_df, :] = spec_f.unsqueeze(1)
        return out

    def forward(self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor):
        # feat_spec supports [B,1,T,F,2] or [B,2,T,F]
        if feat_spec.dim() == 5:
            feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        elif feat_spec.dim() == 4 and feat_spec.shape[1] == 1 and feat_spec.shape[-1] == 2:
            feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        elif feat_spec.dim() == 4 and feat_spec.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Unsupported feat_spec shape: {tuple(feat_spec.shape)}")

        feat_erb = self.pad_feat(feat_erb)
        feat_spec = self.pad_feat(feat_spec)

        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)

        if self.run_erb:
            m = self.erb_dec(emb, e3, e2, e1, e0)
            spec_m = self.mask(spec, m)
        else:
            m = torch.zeros((spec.shape[0], 1, spec.shape[2], self.erb_bins), device=spec.device, dtype=spec.dtype)
            spec_m = torch.zeros_like(spec)

        if self.run_df:
            df_coefs_raw = self.df_dec(emb, c0)
            df_coefs = self.df_out_transform(df_coefs_raw)
            spec_e = self.apply_df(spec.clone(), df_coefs)
            spec_e[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_e = spec_m

        if self.post_filter:
            beta = self.post_filter_beta
            eps = 1e-12
            spec_e_c = as_complex(spec_e)
            spec_c = as_complex(spec)
            mask = (spec_e_c.abs() / spec_c.abs().add(eps)).clamp(eps, 1)
            mask_sin = mask * torch.sin(math.pi * mask / 2).clamp_min(eps)
            pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2))
            spec_e = spec_e * pf.unsqueeze(-1)

        sep_mask = self.sep_head(emb)
        return spec_e, m, lsnr, df_coefs, sep_mask


class DfNetSE(nn.Module):
    """Waveform-friendly wrapper around DfNet."""

    def __init__(self, config: Optional[ModelConfig] = None, run_df: bool = True):
        super().__init__()
        self.config = config or get_config()
        widths = compute_erb_widths(
            self.config.sr,
            self.config.fft_size,
            self.config.nb_erb,
            self.config.min_nb_freqs,
        )
        erb = erb_fb(widths, self.config.sr, inverse=False)
        erb_inv = erb_fb(widths, self.config.sr, inverse=True)
        self.model = DfNet(erb, erb_inv, run_df=run_df, train_mask=True, config=self.config)

        self.window: Tensor
        self.register_buffer("window", torch.hann_window(self.config.fft_size))
        alpha = get_norm_alpha(self.config.sr, self.config.hop_size, self.config.norm_tau)
        self.erb_norm = ExponentialUnitNorm(alpha, self.config.nb_erb)
        self.spec_norm = ExponentialUnitNorm(alpha, self.config.nb_df)

    def stft(self, wav: Tensor) -> Tensor:
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        spec = torch.stft(
            wav.squeeze(1),
            n_fft=self.config.fft_size,
            hop_length=self.config.hop_size,
            window=self.window,
            return_complex=True,
        )
        spec = spec.transpose(1, 2).unsqueeze(1)
        return as_real(spec)

    def istft(self, spec: Tensor, length: Optional[int] = None) -> Tensor:
        spec_c = as_complex(spec).squeeze(1).transpose(1, 2)
        wav = torch.istft(
            spec_c,
            n_fft=self.config.fft_size,
            hop_length=self.config.hop_size,
            window=self.window,
            length=length,
        )
        return wav.unsqueeze(1)

    def erb_features(self, spec: Tensor) -> Tensor:
        spec_c = as_complex(spec)
        power = spec_c.abs().square()
        erb_fb_tensor = self.model.erb_fb
        assert isinstance(erb_fb_tensor, Tensor)
        erb = torch.matmul(power, erb_fb_tensor)
        erb = 10.0 * torch.log10(erb.clamp_min(1e-10))
        return self.erb_norm(erb)

    def spec_features(self, spec: Tensor) -> Tensor:
        spec_df = spec[..., : self.config.nb_df, :]
        spec_df = self.spec_norm(spec_df)
        return spec_df.squeeze(1).permute(0, 3, 1, 2)

    def forward(
        self,
        mixture: Tensor,
        feat_erb: Optional[Tensor] = None,
        feat_spec: Optional[Tensor] = None,
        return_spec: bool = False,
        return_sep_mask: bool = False,
    ):
        mixture_len: Optional[int] = None
        if mixture.dim() == 5:
            spec = mixture
        else:
            if mixture.dim() == 2:
                mixture = mixture.unsqueeze(1)
            mixture_len = mixture.shape[-1]
            spec = self.stft(mixture)

        if feat_erb is None:
            feat_erb = self.erb_features(spec)
        if feat_spec is None:
            feat_spec = self.spec_features(spec)

        spec_e, m, lsnr, df_coefs, sep_mask = self.model(spec, feat_erb, feat_spec)

        if return_spec:
            if return_sep_mask:
                return spec_e, m, sep_mask, lsnr, df_coefs
            return spec_e, lsnr

        enhanced = self.istft(spec_e, length=mixture_len)
        if return_sep_mask:
            return enhanced, m, sep_mask, lsnr
        return enhanced


def _to_model_config(obj) -> ModelConfig:
    if isinstance(obj, ModelConfig):
        return obj
    if hasattr(obj, "df") and hasattr(obj, "model"):
        return ModelConfig(
            sr=obj.df.sr,
            fft_size=obj.df.fft_size,
            hop_size=obj.df.hop_size,
            nb_erb=obj.df.nb_erb,
            nb_df=obj.df.nb_df,
            norm_tau=obj.df.norm_tau,
            lsnr_max=obj.df.lsnr_max,
            lsnr_min=obj.df.lsnr_min,
            min_nb_freqs=obj.df.min_nb_freqs,
            df_order=obj.df.df_order,
            df_lookahead=obj.df.df_lookahead,
            conv_lookahead=obj.model.conv_lookahead,
            conv_ch=obj.model.conv_ch,
            conv_depthwise=obj.model.conv_depthwise,
            convt_depthwise=obj.model.convt_depthwise,
            conv_kernel=tuple(obj.model.conv_kernel),
            convt_kernel=tuple(obj.model.convt_kernel),
            conv_kernel_inp=tuple(obj.model.conv_kernel_inp),
            emb_hidden_dim=obj.model.emb_hidden_dim,
            emb_num_layers=obj.model.emb_num_layers,
            emb_gru_skip_enc=obj.model.emb_gru_skip_enc,
            emb_gru_skip=obj.model.emb_gru_skip,
            df_hidden_dim=obj.model.df_hidden_dim,
            df_gru_skip=obj.model.df_gru_skip,
            df_pathway_kernel_size_t=obj.model.df_pathway_kernel_size_t,
            enc_concat=obj.model.enc_concat,
            df_num_layers=obj.model.df_num_layers,
            df_n_iter=obj.model.df_n_iter,
            lin_groups=obj.model.lin_groups,
            enc_lin_groups=obj.model.enc_lin_groups,
            mask_pf=obj.model.mask_pf,
            pf_beta=obj.model.pf_beta,
            lsnr_dropout=obj.model.lsnr_dropout,
        )
    return get_config()


def create_model(config: Optional[object] = None) -> DfNetSE:
    cfg = _to_model_config(config) if config is not None else get_config()
    return DfNetSE(cfg)


if __name__ == "__main__":
    model = DfNetSE()
    x = torch.randn(2, 1, 16000)
    y = model(x)
    print(f"Input: {tuple(x.shape)}, Output: {tuple(y.shape)}")
