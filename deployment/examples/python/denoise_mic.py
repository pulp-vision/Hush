#!/usr/bin/env python3
"""Real-time microphone denoising demo using Weya NC Standalone.

Captures audio from your microphone, denoises each frame with the
libweya_nc library, and plays the cleaned audio through your speakers.

Requirements::

    pip install sounddevice numpy

Usage::

    python3 denoise_mic.py \\
        --lib deployment/lib/libweya_nc.so \\
        --model deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave

import numpy as np

from weya_nc import WeyaNC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time microphone denoising with Weya NC."
    )
    parser.add_argument("--lib", default=None, help="Path to shared library")
    parser.add_argument("--model", default=None, help="Path to ONNX model bundle")
    parser.add_argument(
        "--atten-lim-db", type=float, default=100.0,
        help="Max attenuation in dB (default: 100 = unlimited)",
    )
    parser.add_argument(
        "--passthrough", action="store_true",
        help="Bypass denoising (A/B comparison)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument(
        "--save-input", default=None, metavar="PATH",
        help="Save raw microphone input to a WAV file",
    )
    parser.add_argument(
        "--save-output", default=None, metavar="PATH",
        help="Save denoised output to a WAV file",
    )
    return parser.parse_args()


def _save_recording(
    path: str | None,
    frames: list[np.ndarray] | None,
    sr: int,
    label: str,
) -> None:
    """Write collected float32 frames to a 16-bit mono WAV file."""
    if path is None or frames is None or len(frames) == 0:
        return
    audio = np.concatenate(frames)
    pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"  Saved {label} recording: {path} ({len(audio) / sr:.1f}s)")


def main() -> None:
    args = parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Error: sounddevice is required for the microphone demo.\n"
            "Install it with: pip install sounddevice",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list_devices:
        print(sd.query_devices())
        return

    nc = WeyaNC(
        lib_path=args.lib,
        model_path=args.model,
        sample_rate=16000,
        atten_lim_db=args.atten_lim_db,
    )

    frame_len = nc.frame_length
    sr = nc.sample_rate
    passthrough = args.passthrough

    # Recording buffers (collect frames, write to disk on exit)
    rec_input: list[np.ndarray] = [] if args.save_input else None
    rec_output: list[np.ndarray] = [] if args.save_output else None

    # SNR tracking
    snr_accum: list[float] = []
    last_print = time.monotonic()

    def callback(
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        nonlocal last_print, snr_accum

        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)

        # indata is (frames, 1) float32
        frame_in = indata[:, 0].copy()

        if passthrough:
            outdata[:, 0] = frame_in
        else:
            outdata[:, 0] = nc.process_frame(frame_in)

        if rec_input is not None:
            rec_input.append(frame_in.copy())
        if rec_output is not None:
            rec_output.append(outdata[:, 0].copy())

        # Simple SNR estimate: ratio of output power to (input - output) power
        sig_power = np.mean(outdata[:, 0] ** 2)
        noise_power = np.mean((frame_in - outdata[:, 0]) ** 2)
        if noise_power > 1e-10:
            snr_db = 10 * np.log10(sig_power / noise_power + 1e-10)
            snr_accum.append(snr_db)

        now = time.monotonic()
        if now - last_print >= 1.0 and snr_accum:
            avg_snr = np.mean(snr_accum)
            print(f"  SNR estimate: {avg_snr:+.1f} dB", end="\r")
            snr_accum.clear()
            last_print = now

    mode = "PASSTHROUGH" if passthrough else "DENOISING"
    print(f"Microphone demo — {mode} mode")
    print(f"  Sample rate: {sr} Hz | Frame: {frame_len} samples ({frame_len / sr * 1000:.0f} ms)")
    if args.save_input:
        print(f"  Recording input  -> {args.save_input}")
    if args.save_output:
        print(f"  Recording output -> {args.save_output}")
    print("  Press Ctrl+C to stop.\n")

    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=frame_len,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        if "PortAudio" in type(exc).__name__ or "PortAudio" in str(exc):
            print(
                f"Audio device error: {exc}\n"
                "Use --list-devices to see available devices.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    finally:
        nc.close()
        _save_recording(args.save_input, rec_input, sr, "input")
        _save_recording(args.save_output, rec_output, sr, "output")


if __name__ == "__main__":
    main()
