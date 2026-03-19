"""
Configuration for DeepFilterNet-SE training.

All defaults match DF3 except:
  1. 16 kHz sample rate with DF3-equivalent time scales (fft_size=320, hop_size=160, nb_df=64)
  2. conv_ch=16 (DF3 default)
  3. max_epochs=100 (user preference)
  4. Canonical real-time profile defaults to nb_erb=32 / nb_df=64
"""

from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class DFConfig:
    """DF parameters – matches DfParams from df/config.py."""
    sr: int = 16000
    fft_size: int = 320
    hop_size: int = 160
    nb_erb: int = 32
    nb_df: int = 64
    df_order: int = 5
    df_lookahead: int = 0
    norm_tau: float = 1.0
    lsnr_max: int = 35
    lsnr_min: int = -15
    min_nb_freqs: int = 2  # Enforce minimum 2 bins per ERB band
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
    """Training configuration.

    Mostly DF3-aligned, with intentional defaults for this project:
    - max_epochs=100
    - batch_size=8
    - prefetch=8
    - p_interfer_sp=0.6
    - p_fill_speech=0.0
    """
    seed: int = 42
    max_epochs: int = 100  # User preference (DF3 default is 10)
    batch_size: int = 8
    batch_size_eval: int = 0
    num_workers: int = 4
    max_sample_len_s: float = 5.0
    log_freq: int = 100
    prefetch: int = 8
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
    p_interfer_sp: float = 0.6  # Include background speaker mixing for most, not all, samples
    p_fill_speech: float = 0.0  # Keep primary target single-speaker by default
    snrs: List[int] = field(default_factory=lambda: [-5, 0, 5, 10, 20, 40])
    gains: List[int] = field(default_factory=lambda: [-6, 0, 6])
    # Keep background speaker consistently quieter than primary speech.
    interfer_snrs: List[float] = field(default_factory=lambda: [24.0, 20.0, 18.0, 16.0, 14.0, 12.0])
    global_sampling_factor: float = 1.0


@dataclass
class MlflowConfig:
    """MLflow tracking configuration."""
    enabled: bool = True
    tracking_uri: str = ""
    experiment_name: str = "AdvancedDeepFilterNet"
    run_name: str = ""
    log_every_n_steps: int = 100
    log_artifacts: bool = True


@dataclass
class Config:
    """Complete configuration."""
    df: DFConfig = field(default_factory=DFConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)

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


def _parse_csv(raw: str, cast: Callable[[str], T]) -> List[T]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def load_config_from_ini(path: str, base: Optional[Config] = None) -> Config:
    """Load config values from an INI file into the dataclass configuration."""
    parser = ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    cfg = base if base is not None else get_default_config()

    def has(section: str, option: str) -> bool:
        return parser.has_section(section) and parser.has_option(section, option)

    def set_i(section: str, option: str, obj, attr: str) -> None:
        if has(section, option):
            setattr(obj, attr, parser.getint(section, option))

    def set_f(section: str, option: str, obj, attr: str) -> None:
        if has(section, option):
            setattr(obj, attr, parser.getfloat(section, option))

    def set_b(section: str, option: str, obj, attr: str) -> None:
        if has(section, option):
            setattr(obj, attr, parser.getboolean(section, option))

    def set_s(section: str, option: str, obj, attr: str) -> None:
        if has(section, option):
            setattr(obj, attr, parser.get(section, option))

    def set_csv(section: str, option: str, obj, attr: str, cast: Callable[[str], T]) -> None:
        if has(section, option):
            setattr(obj, attr, _parse_csv(parser.get(section, option), cast))

    def set_pair_i(section: str, option: str, obj, attr: str) -> None:
        if has(section, option):
            vals = _parse_csv(parser.get(section, option), int)
            if len(vals) >= 2:
                setattr(obj, attr, (vals[0], vals[1]))

    # [df]
    set_i("df", "sr", cfg.df, "sr")
    set_i("df", "fft_size", cfg.df, "fft_size")
    set_i("df", "hop_size", cfg.df, "hop_size")
    set_i("df", "nb_erb", cfg.df, "nb_erb")
    set_i("df", "nb_df", cfg.df, "nb_df")
    set_i("df", "min_nb_erb_freqs", cfg.df, "min_nb_freqs")
    set_i("df", "df_order", cfg.df, "df_order")
    set_i("df", "df_lookahead", cfg.df, "df_lookahead")
    set_i("df", "lsnr_max", cfg.df, "lsnr_max")
    set_i("df", "lsnr_min", cfg.df, "lsnr_min")
    set_f("df", "norm_tau", cfg.df, "norm_tau")

    # [train]
    set_i("train", "seed", cfg.train, "seed")
    set_i("train", "max_epochs", cfg.train, "max_epochs")
    set_i("train", "batch_size", cfg.train, "batch_size")
    set_i("train", "batch_size_eval", cfg.train, "batch_size_eval")
    set_i("train", "num_workers", cfg.train, "num_workers")
    set_i("train", "prefetch", cfg.train, "prefetch")
    set_i("train", "log_freq", cfg.train, "log_freq")
    set_i("train", "early_stopping_patience", cfg.train, "early_stopping_patience")
    set_f("train", "max_sample_len_s", cfg.train, "max_sample_len_s")
    set_f("train", "grad_clip", cfg.train, "grad_clip")
    set_f("train", "global_ds_sampling_f", cfg.train, "global_sampling_factor")
    set_s("train", "validation_criteria", cfg.train, "validation_criteria")
    set_s("train", "validation_criteria_rule", cfg.train, "validation_criteria_rule")
    set_b("train", "detect_anomaly", cfg.train, "detect_anomaly")
    if has("train", "dataloader_snrs"):
        cfg.train.snrs = _parse_csv(parser.get("train", "dataloader_snrs"), int)
    if has("train", "dataloader_gains"):
        cfg.train.gains = _parse_csv(parser.get("train", "dataloader_gains"), int)

    # [distortion]
    set_f("distortion", "p_reverb", cfg.train, "p_reverb")
    set_f("distortion", "p_bandwidth_ext", cfg.train, "p_bw_ext")
    set_f("distortion", "p_clipping", cfg.train, "p_clipping")
    set_f("distortion", "p_zeroing", cfg.train, "p_zeroing")
    set_f("distortion", "p_air_absorption", cfg.train, "p_air_absorption")
    set_f("distortion", "p_interfer_sp", cfg.train, "p_interfer_sp")
    set_f("distortion", "p_fill_speech", cfg.train, "p_fill_speech")
    if has("distortion", "interfer_snrs"):
        cfg.train.interfer_snrs = _parse_csv(parser.get("distortion", "interfer_snrs"), float)

    # [model] and [deepfilternet] aliases
    for sec in ("model", "deepfilternet"):
        set_i(sec, "conv_lookahead", cfg.model, "conv_lookahead")
        set_i(sec, "conv_ch", cfg.model, "conv_ch")
        set_b(sec, "conv_depthwise", cfg.model, "conv_depthwise")
        set_b(sec, "convt_depthwise", cfg.model, "convt_depthwise")
        set_pair_i(sec, "conv_kernel", cfg.model, "conv_kernel")
        set_pair_i(sec, "convt_kernel", cfg.model, "convt_kernel")
        set_pair_i(sec, "conv_kernel_inp", cfg.model, "conv_kernel_inp")
        set_i(sec, "emb_hidden_dim", cfg.model, "emb_hidden_dim")
        set_i(sec, "emb_num_layers", cfg.model, "emb_num_layers")
        set_s(sec, "emb_gru_skip_enc", cfg.model, "emb_gru_skip_enc")
        set_s(sec, "emb_gru_skip", cfg.model, "emb_gru_skip")
        set_i(sec, "df_hidden_dim", cfg.model, "df_hidden_dim")
        set_s(sec, "df_gru_skip", cfg.model, "df_gru_skip")
        set_i(sec, "df_pathway_kernel_size_t", cfg.model, "df_pathway_kernel_size_t")
        set_b(sec, "enc_concat", cfg.model, "enc_concat")
        set_i(sec, "df_num_layers", cfg.model, "df_num_layers")
        set_i(sec, "df_n_iter", cfg.model, "df_n_iter")
        set_i(sec, "lin_groups", cfg.model, "lin_groups")
        set_i(sec, "enc_lin_groups", cfg.model, "enc_lin_groups")
        set_i(sec, "linear_groups", cfg.model, "lin_groups")
        set_i(sec, "enc_linear_groups", cfg.model, "enc_lin_groups")
        set_b(sec, "mask_pf", cfg.model, "mask_pf")
        set_f(sec, "pf_beta", cfg.model, "pf_beta")
        set_b(sec, "lsnr_dropout", cfg.model, "lsnr_dropout")

    # [optim]
    set_s("optim", "optimizer", cfg.optim, "optimizer")
    set_f("optim", "lr", cfg.optim, "lr")
    set_f("optim", "lr_min", cfg.optim, "lr_min")
    set_f("optim", "lr_warmup", cfg.optim, "lr_warmup")
    set_f("optim", "weight_decay", cfg.optim, "weight_decay")
    set_f("optim", "weight_decay_end", cfg.optim, "weight_decay_end")
    set_b("optim", "amsgrad", cfg.optim, "amsgrad")
    set_i("optim", "warmup_epochs", cfg.optim, "warmup_epochs")
    set_f("optim", "lr_cycle_mul", cfg.optim, "lr_cycle_mul")
    set_f("optim", "lr_cycle_decay", cfg.optim, "lr_cycle_decay")
    set_i("optim", "lr_cycle_epochs", cfg.optim, "lr_cycle_epochs")
    if has("optim", "betas"):
        vals = _parse_csv(parser.get("optim", "betas"), float)
        if len(vals) >= 2:
            cfg.optim.betas = (vals[0], vals[1])

    # [loss] direct mapping
    set_f("loss", "ml_factor", cfg.loss, "ml_factor")
    set_s("loss", "ml_mask", cfg.loss, "ml_mask")
    set_f("loss", "ml_gamma", cfg.loss, "ml_gamma")
    set_f("loss", "ml_gamma_pred", cfg.loss, "ml_gamma_pred")
    set_f("loss", "ml_f_under", cfg.loss, "ml_f_under")
    set_f("loss", "ml_max_freq", cfg.loss, "ml_max_freq")
    set_f("loss", "sl_factor_magnitude", cfg.loss, "sl_factor_magnitude")
    set_f("loss", "sl_factor_complex", cfg.loss, "sl_factor_complex")
    set_f("loss", "sl_factor_under", cfg.loss, "sl_factor_under")
    set_f("loss", "sl_gamma", cfg.loss, "sl_gamma")
    set_f("loss", "mrsl_factor", cfg.loss, "mrsl_factor")
    set_f("loss", "mrsl_factor_complex", cfg.loss, "mrsl_factor_complex")
    set_f("loss", "mrsl_gamma", cfg.loss, "mrsl_gamma")
    set_csv("loss", "mrsl_fft_sizes", cfg.loss, "mrsl_fft_sizes", int)
    set_f("loss", "sdrl_factor", cfg.loss, "sdrl_factor")
    set_csv("loss", "sdrl_segmental_ws", cfg.loss, "sdrl_segmental_ws", int)
    set_f("loss", "lsnr_factor", cfg.loss, "lsnr_factor")
    set_f("loss", "sep_factor", cfg.loss, "sep_factor")

    # DeepFilterNet-compatible loss sections
    set_f("maskloss", "factor", cfg.loss, "ml_factor")
    set_s("maskloss", "mask", cfg.loss, "ml_mask")
    set_f("maskloss", "gamma", cfg.loss, "ml_gamma")
    set_f("maskloss", "gamma_pred", cfg.loss, "ml_gamma_pred")
    set_f("maskloss", "f_under", cfg.loss, "ml_f_under")
    set_f("maskloss", "max_freq", cfg.loss, "ml_max_freq")

    set_f("spectralloss", "factor_magnitude", cfg.loss, "sl_factor_magnitude")
    set_f("spectralloss", "factor_complex", cfg.loss, "sl_factor_complex")
    set_f("spectralloss", "factor_under", cfg.loss, "sl_factor_under")
    set_f("spectralloss", "gamma", cfg.loss, "sl_gamma")

    set_f("multiresspecloss", "factor", cfg.loss, "mrsl_factor")
    set_f("multiresspecloss", "factor_complex", cfg.loss, "mrsl_factor_complex")
    set_f("multiresspecloss", "gamma", cfg.loss, "mrsl_gamma")
    set_csv("multiresspecloss", "fft_sizes", cfg.loss, "mrsl_fft_sizes", int)

    set_f("localsnrloss", "factor", cfg.loss, "lsnr_factor")

    set_f("sdrloss", "factor", cfg.loss, "sdrl_factor")
    set_csv("sdrloss", "segmental_ws", cfg.loss, "sdrl_segmental_ws", int)

    return cfg
