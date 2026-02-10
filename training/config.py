"""
Configuration for DeepFilterNet-SE training.

All defaults match DF3 except:
  1. 16 kHz sample rate (fft_size=512, hop_size=128, nb_df=64)
  2. conv_ch=16 (DF3 default, was incorrectly 64)
  3. max_epochs=100 (user preference)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DFConfig:
    """DF parameters – matches DfParams from df/config.py."""
    sr: int = 16000
    fft_size: int = 512
    hop_size: int = 128
    nb_erb: int = 32
    nb_df: int = 64
    df_order: int = 5
    df_lookahead: int = 0
    norm_tau: float = 1.0
    lsnr_max: int = 35
    lsnr_min: int = -15
    min_nb_freqs: int = 2
    pad_mode: str = "input"

    @property
    def n_freqs(self) -> int:
        return self.fft_size // 2 + 1


@dataclass
class ModelConfig:
    """Model architecture configuration – matches DF3 ModelParams."""
    conv_ch: int = 16
    conv_depthwise: bool = True
    convt_depthwise: bool = True
    conv_kernel: Tuple[int, int] = (1, 3)
    convt_kernel: Tuple[int, int] = (1, 3)
    conv_kernel_inp: Tuple[int, int] = (3, 3)
    conv_lookahead: int = 0
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


@dataclass
class OptimConfig:
    """Optimizer configuration – matches DF3 defaults."""
    optimizer: str = "adamw"
    lr: float = 5e-4
    lr_min: float = 1e-6
    lr_warmup: float = 1e-4
    weight_decay: float = 0.05
    weight_decay_end: float = -1.0  # -1 = no WD scheduling
    betas: Tuple[float, float] = (0.9, 0.999)
    amsgrad: bool = True
    warmup_epochs: int = 3
    lr_cycle_mul: float = 1.0
    lr_cycle_decay: float = 0.5
    lr_cycle_epochs: int = -1


@dataclass
class LossConfig:
    """Loss weights – matches DF3 defaults."""
    # MaskLoss
    ml_factor: float = 1.0
    ml_mask: str = "iam"
    ml_gamma: float = 0.6
    ml_gamma_pred: float = 0.6
    ml_f_under: float = 2.0
    ml_max_freq: float = 0.0

    # SpectralLoss
    sl_factor_magnitude: float = 1e4
    sl_factor_complex: float = 0.0
    sl_factor_under: float = 1.0
    sl_gamma: float = 1.0

    # MultiResSpecLoss
    mrsl_factor: float = 0.0
    mrsl_factor_complex: float = 0.0
    mrsl_gamma: float = 1.0
    mrsl_fft_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])

    # SdrLoss
    sdrl_factor: float = 0.0
    sdrl_segmental_ws: List[int] = field(default_factory=list)

    # LocalSnrLoss
    lsnr_factor: float = 0.0005

    # SeparationLoss (intentional addition)
    sep_factor: float = 0.2


@dataclass
class TrainConfig:
    """Training configuration – matches DF3 defaults."""
    seed: int = 42
    max_epochs: int = 100  # User preference (DF3 default is 10)
    batch_size: int = 96
    batch_size_eval: int = 0  # 0 = same as batch_size
    num_workers: int = 4
    max_sample_len_s: float = 5.0  # DF3 default
    log_freq: int = 100
    prefetch: int = 32  # DF3 default
    grad_clip: float = 1.0
    detect_anomaly: bool = False

    # Validation / early stopping
    validation_criteria: str = "loss"
    validation_criteria_rule: str = "min"
    early_stopping_patience: int = 5

    # Data augmentation
    p_reverb: float = 0.2
    p_bw_ext: float = 0.0
    p_clipping: float = 0.0
    p_zeroing: float = 0.0
    p_air_absorption: float = 0.0
    p_interfer_sp: float = 1.0  # Intentional: always include background speaker mixing
    snrs: List[int] = field(default_factory=lambda: [-5, 0, 5, 10, 20, 40])
    gains: List[int] = field(default_factory=lambda: [-6, 0, 6])


@dataclass
class Config:
    """Complete configuration."""
    df: DFConfig = field(default_factory=DFConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Paths
    base_dir: str = "."
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda"


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def config_to_dict(config: Config) -> dict:
    from dataclasses import asdict
    return asdict(config)
