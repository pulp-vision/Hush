"""
Audio augmentation utilities for DeepFilterNet-SE.

Implements synthetic mixing of primary + background speakers + noise.
The target is always the clean primary speaker, enabling the model to learn
joint denoising and background speaker suppression.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor

try:
    from scipy.signal import fftconvolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def load_audio(path: str, target_sr: int = 16000) -> Tensor:
    """Load audio file and resample to target sample rate."""
    if HAS_SOUNDFILE:
        # Use soundfile for more robust loading
        try:
            audio, sr = sf.read(str(path), dtype='float32')
            
            # Convert to tensor
            if len(audio.shape) == 1:
                audio = audio[np.newaxis, :]  # [1, T]
            else:
                audio = audio.T  # [C, T]
                audio = audio.mean(axis=0, keepdims=True)  # Mono
            
            waveform = torch.from_numpy(audio)
            
            # Resample if needed
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
            return waveform
        except Exception:
            pass  # Fall through to torchaudio
    
    # Fallback to torchaudio
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def compute_rms(waveform: Tensor, eps: float = 1e-8) -> float:
    """Compute RMS energy of waveform."""
    return float(torch.sqrt(torch.mean(waveform**2) + eps))


def normalize_loudness(waveform: Tensor, target_lufs: float = -23.0) -> Tensor:
    """Normalize waveform to target loudness (simplified LUFS approximation)."""
    # Simplified: use RMS-based normalization as LUFS approximation
    current_rms = compute_rms(waveform)
    if current_rms < 1e-8:
        return waveform
    # Target RMS for -23 LUFS is approximately 0.05
    target_rms = 10 ** ((target_lufs + 23) / 20) * 0.05
    gain = target_rms / current_rms
    return waveform * gain


def pitch_shift(
    waveform: Tensor, 
    sample_rate: int, 
    semitones: float,
    n_fft: int = 512,
) -> Tensor:
    """Shift pitch by specified semitones."""
    # Use torchaudio's pitch_shift if available (torch >= 2.0)
    try:
        shifted = torchaudio.functional.pitch_shift(
            waveform, 
            sample_rate, 
            semitones,
            n_fft=n_fft,
        )
    except Exception:
        # Fallback: simple resampling-based pitch shift (less accurate)
        ratio = 2 ** (semitones / 12)
        resampled = torchaudio.functional.resample(
            waveform, 
            sample_rate, 
            int(sample_rate / ratio)
        )
        # Resample back to original rate
        shifted = torchaudio.functional.resample(
            resampled,
            int(sample_rate / ratio),
            sample_rate
        )
    return shifted


def mix_signals(
    primary: Tensor,
    secondary: Optional[Tensor] = None,
    noise: Optional[Tensor] = None,
    secondary_scale: float = 0.4,  # Scale relative to primary RMS
    noise_snr_db: float = 10.0,    # SNR for noise addition
) -> Tensor:
    """
    Mix primary speaker with optional secondary speaker and noise.
    
    Args:
        primary: Primary speaker waveform [1, T]
        secondary: Background speaker waveform [1, T] (will be scaled down)
        noise: Noise waveform [1, T]
        secondary_scale: Scale factor for secondary speaker (0.2-0.6 typical)
        noise_snr_db: Target SNR for noise addition
    
    Returns:
        Mixed waveform [1, T]
    """
    mixture = primary.clone()
    primary_rms = compute_rms(primary)
    
    # Add secondary speaker (background) at lower amplitude
    if secondary is not None and secondary.numel() > 0:
        # Match length
        if secondary.shape[-1] > primary.shape[-1]:
            start = random.randint(0, secondary.shape[-1] - primary.shape[-1])
            secondary = secondary[..., start:start + primary.shape[-1]]
        elif secondary.shape[-1] < primary.shape[-1]:
            # Pad or loop
            repeats = (primary.shape[-1] // secondary.shape[-1]) + 1
            secondary = secondary.repeat(1, repeats)[..., :primary.shape[-1]]
        
        # Scale secondary relative to primary
        secondary_rms = compute_rms(secondary)
        if secondary_rms > 1e-8:
            scale = (primary_rms * secondary_scale) / secondary_rms
            mixture = mixture + secondary * scale
    
    # Add noise at specified SNR
    if noise is not None and noise.numel() > 0:
        # Match length
        if noise.shape[-1] > mixture.shape[-1]:
            start = random.randint(0, noise.shape[-1] - mixture.shape[-1])
            noise = noise[..., start:start + mixture.shape[-1]]
        elif noise.shape[-1] < mixture.shape[-1]:
            repeats = (mixture.shape[-1] // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)[..., :mixture.shape[-1]]
        
        # Scale noise to target SNR
        noise_rms = compute_rms(noise)
        if noise_rms > 1e-8:
            target_noise_rms = primary_rms / (10 ** (noise_snr_db / 20))
            scale = target_noise_rms / noise_rms
            mixture = mixture + noise * scale
    
    return mixture


class MixingAugmentor:
    """
    Augmentor for creating synthetic speaker mixtures.
    
    Creates mixtures of:
    - Primary speaker (target, unmodified)
    - Background speaker (OPTIONALLY pitch-shifted, lower amplitude)
    - Noise (DNS-style)
    
    IMPORTANT: Pitch shifting is only applied to ~40% of samples to ensure
    the model learns to distinguish speakers by VOLUME and TIMBRE, not by
    artificial pitch artifacts. This improves real-world generalization.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        secondary_scale_range: Tuple[float, float] = (0.15, 0.5),
        pitch_shift_range: Tuple[float, float] = (-4.0, 4.0),  # Bidirectional
        noise_snr_range: Tuple[float, float] = (5.0, 20.0),
        p_secondary: float = 0.8,  # Probability of adding secondary speaker
        p_noise: float = 0.9,      # Probability of adding noise
        p_pitch_shift: float = 0.4,  # Only 40% of secondary speakers get pitch-shifted
        p_reverb: float = 0.2,     # Probability of adding reverb (matching DFN3)
        rir_paths: Optional[list] = None,  # List of RIR file paths
    ):
        self.sample_rate = sample_rate
        self.secondary_scale_range = secondary_scale_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_snr_range = noise_snr_range
        self.p_secondary = p_secondary
        self.p_noise = p_noise
        self.p_pitch_shift = p_pitch_shift
        self.p_reverb = p_reverb
        self.rir_paths = rir_paths or []
    
    def __call__(
        self,
        primary: Tensor,
        secondary: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Create augmented mixture.
        
        Args:
            primary: Primary speaker waveform [1, T]
            secondary: Background speaker waveform [1, T]
            noise: Noise waveform [1, T]
        
        Returns:
            Tuple of (mixture, clean_primary)
        """
        # Normalize primary speaker
        primary = normalize_loudness(primary)
        
        # Apply reverb to primary (matching DFN3 p_reverb=0.2)
        if HAS_SCIPY and self.rir_paths and random.random() < self.p_reverb:
            rir_path = random.choice(self.rir_paths)
            try:
                rir = load_audio(rir_path, self.sample_rate)
                primary_np = primary.squeeze().numpy()
                rir_np = rir.squeeze().numpy()
                # Convolve and normalize
                primary_conv = fftconvolve(primary_np, rir_np, mode='same')
                max_val = np.abs(primary_conv).max()
                if max_val > 1e-8:
                    primary_conv = primary_conv / max_val * 0.9
                primary = torch.from_numpy(primary_conv).unsqueeze(0).float()
            except Exception:
                pass  # Skip reverb if RIR loading fails
        
        clean_primary = primary.clone()
        
        # Optionally add secondary speaker
        processed_secondary = None
        if secondary is not None and random.random() < self.p_secondary:
            processed_secondary = secondary
            
            # OPTIONAL pitch shift (only ~40% of the time)
            # Primary distinction is VOLUME, not pitch
            if random.random() < self.p_pitch_shift:
                pitch_semitones = random.uniform(*self.pitch_shift_range)
                processed_secondary = pitch_shift(
                    processed_secondary, 
                    self.sample_rate, 
                    pitch_semitones
                )
            
        # Optionally add noise
        processed_noise = None
        if noise is not None and random.random() < self.p_noise:
            processed_noise = noise
        
        # Sample augmentation parameters
        secondary_scale = random.uniform(*self.secondary_scale_range)
        noise_snr = random.uniform(*self.noise_snr_range)
        
        # Create mixture
        mixture = mix_signals(
            primary=primary,
            secondary=processed_secondary,
            noise=processed_noise,
            secondary_scale=secondary_scale,
            noise_snr_db=noise_snr,
        )
        
        # Clip to prevent overflow
        mixture = torch.clamp(mixture, -1.0, 1.0)
        clean_primary = torch.clamp(clean_primary, -1.0, 1.0)
        
        return mixture, clean_primary


def random_crop(waveform: Tensor, length: int) -> Tensor:
    """Randomly crop waveform to specified length."""
    if waveform.shape[-1] <= length:
        # Pad if too short
        pad_size = length - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        start = random.randint(0, waveform.shape[-1] - length)
        waveform = waveform[..., start:start + length]
    return waveform


def time_stretch(waveform: Tensor, rate: float, sample_rate: int = 16000) -> Tensor:
    """Apply time stretching (speed change without pitch change)."""
    # Simple implementation via resampling
    stretched = torchaudio.functional.resample(
        waveform,
        sample_rate,
        int(sample_rate * rate)
    )
    return torchaudio.functional.resample(
        stretched,
        int(sample_rate * rate),
        sample_rate
    )
