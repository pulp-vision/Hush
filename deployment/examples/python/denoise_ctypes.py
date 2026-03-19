#!/usr/bin/env python3
"""Minimal external denoising example for libweya_nc via ctypes.

Usage:
  python3 denoise_ctypes.py \
    --lib /path/to/libweya_nc.so \
    --model /path/to/advanced_dfnet16k_model_best_onnx.tar.gz \
    --input noisy.wav \
    --output denoised.wav

The library file varies by platform:
  Linux:   libweya_nc.so
  macOS:   libweya_nc.dylib
  Windows: weya_nc.dll
"""

from __future__ import annotations

import argparse
import ctypes
import platform
import wave
from pathlib import Path

import numpy as np


def get_default_lib_name() -> str:
    """Return the platform-appropriate library filename."""
    system = platform.system()
    if system == "Darwin":
        return "libweya_nc.dylib"
    elif system == "Windows":
        return "weya_nc.dll"
    return "libweya_nc.so"


def load_wav_mono_i16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM wav, got sample width={sampwidth}")
    audio = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif channels != 1:
        raise ValueError(f"Expected mono/stereo wav, got channels={channels}")
    return audio, sr


def write_wav_mono_i16(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.astype(np.int16).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Denoise a WAV file using libweya_nc."
    )
    parser.add_argument(
        "--lib", type=Path, required=True,
        help=f"Path to shared library (default name: {get_default_lib_name()})",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atten-lim-db", type=float, default=100.0)
    args = parser.parse_args()

    lib = ctypes.CDLL(str(args.lib.resolve()))

    lib.weya_nc_model_load_from_path.argtypes = [ctypes.c_char_p]
    lib.weya_nc_model_load_from_path.restype = ctypes.c_void_p
    lib.weya_nc_model_free.argtypes = [ctypes.c_void_p]
    lib.weya_nc_model_free.restype = None
    lib.weya_nc_session_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float]
    lib.weya_nc_session_create.restype = ctypes.c_void_p
    lib.weya_nc_session_free.argtypes = [ctypes.c_void_p]
    lib.weya_nc_session_free.restype = None
    lib.weya_nc_get_frame_length.argtypes = [ctypes.c_void_p]
    lib.weya_nc_get_frame_length.restype = ctypes.c_size_t
    lib.weya_nc_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.weya_nc_process_frame.restype = ctypes.c_float

    model = lib.weya_nc_model_load_from_path(str(args.model.resolve()).encode("utf-8"))
    if not model:
        raise RuntimeError(f"Could not load model: {args.model}")

    audio_i16, sr = load_wav_mono_i16(args.input.resolve())
    session = lib.weya_nc_session_create(model, sr, ctypes.c_float(args.atten_lim_db))
    if not session:
        lib.weya_nc_model_free(model)
        raise RuntimeError("Could not create session")

    frame_len = int(lib.weya_nc_get_frame_length(session))
    frame_in = np.zeros(frame_len, dtype=np.float32)
    frame_out = np.zeros(frame_len, dtype=np.float32)
    out_i16 = np.zeros_like(audio_i16, dtype=np.int16)

    idx = 0
    while idx < len(audio_i16):
        end = min(idx + frame_len, len(audio_i16))
        chunk = audio_i16[idx:end].astype(np.float32) / 32768.0
        frame_in.fill(0.0)
        frame_in[: len(chunk)] = chunk

        lib.weya_nc_process_frame(
            session,
            frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        out_chunk = (frame_out[: len(chunk)] * 32768.0).clip(-32768, 32767).astype(np.int16)
        out_i16[idx:end] = out_chunk
        idx = end

    write_wav_mono_i16(args.output.resolve(), out_i16, sr)
    lib.weya_nc_session_free(session)
    lib.weya_nc_model_free(model)

    print(f"Saved denoised audio: {args.output.resolve()}")


if __name__ == "__main__":
    main()
