#!/usr/bin/env python3
"""WebSocket streaming denoising pipeline using Weya NC Standalone.

Shows how to integrate the denoising library into a chunk-based audio
pipeline over WebSockets.

**Server mode** (default): accepts WebSocket connections, receives raw
int16 PCM chunks (16 kHz mono), denoises frame-by-frame, and sends back
clean int16 PCM.  Each connection gets its own session.

**Client mode**: reads a WAV file, streams it over WebSocket, collects
the denoised output, and saves to a new WAV file.

Requirements::

    pip install websockets numpy

Server::

    python3 denoise_stream.py \\
        --lib deployment/lib/libweya_nc.so \\
        --model deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz

Client (in another terminal)::

    python3 denoise_stream.py --mode client \\
        --input noisy.wav --output denoised.wav
"""

from __future__ import annotations

import argparse
import asyncio
import struct
import sys
import wave
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def _handle_connection(
    websocket: object,
    nc_factory: object,
) -> None:
    """Handle a single WebSocket connection (one denoising session)."""
    from weya_nc import WeyaNC

    nc: WeyaNC = nc_factory()
    frame_len = nc.frame_length
    bytes_per_frame = frame_len * 2  # int16 = 2 bytes per sample

    buf = bytearray()
    frames_processed = 0

    try:
        async for message in websocket:
            if isinstance(message, str):
                # Ignore text messages (could be used for control later)
                continue

            buf.extend(message)

            # Process all complete frames in the buffer
            while len(buf) >= bytes_per_frame:
                raw = bytes(buf[:bytes_per_frame])
                del buf[:bytes_per_frame]

                frame_in = np.frombuffer(raw, dtype=np.int16)
                frame_out = nc.process_frame(frame_in)

                await websocket.send(frame_out.tobytes())
                frames_processed += 1
    finally:
        nc.close()
        print(f"  Connection closed — {frames_processed} frames processed")


async def run_server(args: argparse.Namespace) -> None:
    try:
        import websockets
    except ImportError:
        print(
            "Error: websockets is required for the streaming server.\n"
            "Install it with: pip install websockets",
            file=sys.stderr,
        )
        sys.exit(1)

    from weya_nc import WeyaNC

    # Validate that lib/model can be loaded once before accepting connections
    probe = WeyaNC(
        lib_path=args.lib,
        model_path=args.model,
        sample_rate=16000,
        atten_lim_db=args.atten_lim_db,
    )
    frame_len = probe.frame_length
    probe.close()

    def nc_factory() -> WeyaNC:
        return WeyaNC(
            lib_path=args.lib,
            model_path=args.model,
            sample_rate=16000,
            atten_lim_db=args.atten_lim_db,
        )

    async def handler(websocket: object) -> None:
        peer = websocket.remote_address
        print(f"  Connection from {peer}")
        await _handle_connection(websocket, nc_factory)

    print(f"WebSocket denoising server")
    print(f"  Listening on ws://{args.host}:{args.port}")
    print(f"  Frame: {frame_len} samples ({frame_len * 2} bytes per chunk, int16)")
    print(f"  Protocol: send raw int16 PCM chunks, receive denoised int16 PCM")
    print(f"  Press Ctrl+C to stop.\n")

    async with websockets.serve(handler, args.host, args.port):
        await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _load_wav_i16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Expected 16-bit WAV, got sample width {sw}")
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif ch != 1:
        raise ValueError(f"Expected mono or stereo WAV, got {ch} channels")
    return audio, sr


def _save_wav_i16(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.astype(np.int16).tobytes())


async def run_client(args: argparse.Namespace) -> None:
    try:
        import websockets
    except ImportError:
        print(
            "Error: websockets is required for the streaming client.\n"
            "Install it with: pip install websockets",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    audio, sr = _load_wav_i16(input_path)
    print(f"Loaded {input_path.name}: {len(audio)} samples, {sr} Hz")

    # Frame size: 160 samples at 16 kHz (10 ms) — matches server default
    frame_len = 160
    bytes_per_frame = frame_len * 2

    collected: list[bytes] = []

    async with websockets.connect(args.uri) as ws:
        # Send all frames, collect responses
        idx = 0
        while idx < len(audio):
            end = min(idx + frame_len, len(audio))
            chunk = audio[idx:end]
            # Pad last frame if needed
            if len(chunk) < frame_len:
                chunk = np.pad(chunk, (0, frame_len - len(chunk)))
            await ws.send(chunk.tobytes())

            response = await ws.recv()
            collected.append(response)
            idx = end

    # Reconstruct output
    out_frames = [np.frombuffer(b, dtype=np.int16) for b in collected]
    out_audio = np.concatenate(out_frames)[: len(audio)]

    _save_wav_i16(output_path, out_audio, sr)
    print(f"Saved denoised audio: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebSocket streaming denoising pipeline."
    )
    parser.add_argument(
        "--mode", choices=["server", "client"], default="server",
        help="Run as server (default) or client",
    )

    # Server args
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--lib", default=None, help="Path to shared library")
    parser.add_argument("--model", default=None, help="Path to ONNX model bundle")
    parser.add_argument(
        "--atten-lim-db", type=float, default=100.0,
        help="Max attenuation in dB (default: 100)",
    )

    # Client args
    parser.add_argument("--uri", default="ws://localhost:8765", help="Server URI")
    parser.add_argument("--input", help="Input WAV file (client mode)")
    parser.add_argument("--output", help="Output WAV file (client mode)")

    args = parser.parse_args()

    if args.mode == "client":
        if not args.input or not args.output:
            parser.error("Client mode requires --input and --output")
        asyncio.run(run_client(args))
    else:
        try:
            asyncio.run(run_server(args))
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
