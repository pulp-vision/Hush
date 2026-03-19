# Deployment: Weya NC Standalone Library

The **Weya NC Standalone** library lets you integrate the DeepFilterNet-SE denoising model into any C, C++, or Python application **without PyTorch or any Python ML dependencies**. It is a prebuilt Rust shared library that exposes a clean C ABI and uses ONNX Runtime internally.

Prebuilt binaries are available for **Linux**, **macOS**, and **Windows**.

---

## Downloads (GitHub Releases)

The prebuilt binaries and model bundle are too large to store in git. Download them from the [GitHub Releases](https://github.com/YOUR_USERNAME/deepfilternet-se/releases) page:

| File | Platform | Size | Description |
|---|---|---|---|
| `libweya_nc.so` | Linux x86_64 (glibc 2.31+) | ~23 MB | Shared library |
| `libweya_nc.dylib` | macOS ARM64 (Apple Silicon) | ~18 MB | Shared library |
| `weya_nc.dll` | Windows x86_64 | ~23 MB | Dynamic-link library |
| `advanced_dfnet16k_model_best_onnx.tar.gz` | All platforms | ~8 MB | ONNX model bundle |

After downloading, place them at:

```
deployment/lib/libweya_nc.so          # Linux
deployment/lib/libweya_nc.dylib       # macOS
deployment/lib/weya_nc.dll            # Windows
deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz
```

---

## Python Quickstart

**Requirements:** Python 3.8+, numpy

```python
import ctypes
import platform
import numpy as np

# Load the correct library for your platform
system = platform.system()
if system == "Darwin":
    lib = ctypes.CDLL("deployment/lib/libweya_nc.dylib")
elif system == "Windows":
    lib = ctypes.CDLL("deployment/lib/weya_nc.dll")
else:
    lib = ctypes.CDLL("deployment/lib/libweya_nc.so")

# Configure function signatures
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

# Load model and create session
model_path = b"deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz"
model = lib.weya_nc_model_load_from_path(model_path)
session = lib.weya_nc_session_create(model, 16000, ctypes.c_float(100.0))
frame_len = int(lib.weya_nc_get_frame_length(session))

# Process audio frame by frame (float32, range [-1.0, 1.0])
noisy_audio = np.random.randn(frame_len).astype(np.float32) * 0.1  # replace with real audio
frame_out = np.zeros(frame_len, dtype=np.float32)

lib.weya_nc_process_frame(
    session,
    noisy_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
)

# Clean up
lib.weya_nc_session_free(session)
lib.weya_nc_model_free(model)
```

For a complete WAV-file example, see [`examples/python/denoise_ctypes.py`](examples/python/denoise_ctypes.py):

```bash
python3 deployment/examples/python/denoise_ctypes.py \
    --lib deployment/lib/libweya_nc.so \
    --model deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz \
    --input noisy.wav \
    --output denoised.wav
```

The example auto-detects your platform if `--lib` points to a directory containing the library.

---

## Real-Time Examples

Both examples use the shared Python wrapper [`examples/python/weya_nc.py`](examples/python/weya_nc.py), which handles library/model discovery, all 10 C API function declarations, pre-allocated buffers, and int16/float32 conversion.

### Microphone Demo

Captures audio from your microphone, denoises each 10 ms frame, and plays the cleaned audio through your speakers in real time.

**Install:**

```bash
pip install sounddevice numpy
```

**Run:**

```bash
python3 deployment/examples/python/denoise_mic.py \
    --lib deployment/lib/libweya_nc.so \
    --model deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz
```

**Options:**

| Flag | Description |
|---|---|
| `--lib PATH` | Path to shared library (auto-discovered if omitted) |
| `--model PATH` | Path to ONNX model bundle (auto-discovered if omitted) |
| `--atten-lim-db N` | Max attenuation in dB (default: 100 = unlimited) |
| `--passthrough` | Bypass denoising for A/B comparison |
| `--list-devices` | List available audio devices and exit |

The demo prints an SNR estimate once per second so you can gauge the denoising effect. Press Ctrl+C to stop.

### WebSocket Streaming Pipeline

Shows integration into a chunk-based audio pipeline over WebSockets.

**Install:**

```bash
pip install websockets numpy
```

**Server** — receives raw int16 PCM chunks at 16 kHz mono, denoises frame-by-frame, sends back clean int16 PCM. Each WebSocket connection gets its own denoising session.

```bash
python3 deployment/examples/python/denoise_stream.py \
    --lib deployment/lib/libweya_nc.so \
    --model deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz \
    --host 0.0.0.0 --port 8765
```

**Client** — reads a WAV file, streams it over WebSocket, collects the denoised output, saves to a new WAV file. Makes the example fully testable without external tools.

```bash
python3 deployment/examples/python/denoise_stream.py \
    --mode client \
    --uri ws://localhost:8765 \
    --input noisy.wav \
    --output denoised.wav
```

**Protocol:**

- Connect via WebSocket to `ws://host:port`
- Send binary messages containing raw signed int16 PCM (16 kHz, mono)
- Chunks are buffered server-side; a denoised int16 frame (320 bytes = 160 samples) is sent back for each complete frame received
- Each connection is independent with its own session state

---

## C Quickstart

### Linux

```bash
gcc -o denoise_demo \
    deployment/examples/c/denoise_frame_demo.c \
    -Ideployment/include \
    -Ldeployment/lib \
    -lweya_nc \
    -Wl,-rpath,deployment/lib
```

### macOS

```bash
clang -o denoise_demo \
    deployment/examples/c/denoise_frame_demo.c \
    -Ideployment/include \
    -Ldeployment/lib \
    -lweya_nc \
    -Wl,-rpath,@loader_path/deployment/lib
```

### Windows (MinGW)

```bash
gcc -o denoise_demo.exe ^
    deployment\examples\c\denoise_frame_demo.c ^
    -Ideployment\include ^
    -Ldeployment\lib ^
    -lweya_nc
```

Ensure `weya_nc.dll` is in the same directory as the executable or on your `PATH`.

### Run

```bash
./denoise_demo \
    deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz \
    input_i16_mono.pcm \
    output_i16_mono.pcm \
    16000
```

The demo reads/writes raw signed 16-bit PCM (no WAV header). Convert with `sox`:

```bash
# WAV -> raw PCM
sox input.wav -t raw -e signed -b 16 -r 16000 -c 1 input_i16_mono.pcm

# raw PCM -> WAV
sox -t raw -e signed -b 16 -r 16000 -c 1 output_i16_mono.pcm output.wav
```

---

## C API Reference

All functions are declared in [`include/weya_nc.h`](include/weya_nc.h).

### Model lifecycle

| Function | Description |
|---|---|
| `weya_nc_model_load()` | Load model from path in `WEYA_NC_MODEL_PATH` env var |
| `weya_nc_model_load_from_path(path)` | Load ONNX tar.gz bundle from explicit path |
| `weya_nc_model_free(model)` | Free model handle |

### Session lifecycle

| Function | Description |
|---|---|
| `weya_nc_session_create(model, input_sr, atten_lim_db)` | Create streaming session; `input_sr` is your audio sample rate (resampled internally if needed); `atten_lim_db` caps maximum attenuation (100.0 = unlimited) |
| `weya_nc_session_free(session)` | Free session and all associated state |

### Frame processing

| Function | Description |
|---|---|
| `weya_nc_get_frame_length(session)` | Returns the exact number of samples to pass per call (at `input_sr`) |
| `weya_nc_get_sample_rate(session)` | Returns the model's internal sample rate (16000) |
| `weya_nc_get_input_sample_rate(session)` | Returns the configured input sample rate |
| `weya_nc_process_frame(session, input, output)` | Process one frame; input/output are float32 in `[-1.0, 1.0]`; returns estimated SNR in dB |
| `weya_nc_reset(session)` | Reset all GRU/filter state for a new independent audio stream |

### Threading

Each `WeyaSession` is **not thread-safe** — use one session per thread. Multiple sessions can share one `WeyaModel` safely (the model is read-only after load).

---

## Advanced Usage

### Attenuation limit

`atten_lim_db` controls maximum noise suppression. Lower values preserve more background:

```python
# Aggressive suppression (default)
session = lib.weya_nc_session_create(model, 16000, ctypes.c_float(100.0))

# Gentle suppression (preserve some ambience)
session = lib.weya_nc_session_create(model, 16000, ctypes.c_float(20.0))
```

### Multi-stream (e.g. conferencing)

Allocate one session per audio stream. Reset between calls if processing independent streams sequentially:

```python
lib.weya_nc_reset(session)  # clear GRU state before processing a new file
```

### Input sample rates other than 16 kHz

The library resamples internally. Pass your actual input sample rate to `weya_nc_session_create` — `weya_nc_get_frame_length` will return the frame size at that rate:

```python
session = lib.weya_nc_session_create(model, 48000, ctypes.c_float(100.0))
frame_len = int(lib.weya_nc_get_frame_length(session))  # frame size at 48 kHz
```

---

## PyTorch Model vs Prebuilt Library

| | PyTorch model (`model/dfnet_se.py`) | Prebuilt library (`libweya_nc`) |
|---|---|---|
| **Use case** | Training, research, fine-tuning | Production deployment, integration |
| **Dependencies** | Python, PyTorch 2.0+, torchaudio | None (self-contained) |
| **Platforms** | Any (Python) | Linux x86_64, macOS ARM64, Windows x86_64 |
| **Language** | Python | C ABI (usable from C, C++, Python, Go, Rust, ...) |
| **Model format** | `.ckpt` (PyTorch state dict) | ONNX tar.gz bundle |
| **Binary size** | ~1.8M parameters + PyTorch | ~18-23 MB shared lib + ~8 MB model |
| **Latency** | ~20 ms algorithmic (10 ms frame) | ~20 ms algorithmic (10 ms frame) |
| **Real-time** | Yes (with CUDA or fast CPU) | Yes (ONNX Runtime, CPU-optimized) |
| **Fine-tuning** | Yes | No |
