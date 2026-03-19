# DeepFilterNet-SE

**The first open-source speech enhancement model built specifically for Voice AI — with real-time background speaker suppression.**

> **8 MB model · Runs fully on CPU in real time · Trained on 10,000+ hours of mixed audio · Under 1 ms processing per 10 ms of audio**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

---

## Built for Voice AI

DeepFilterNet-SE is designed from the ground up for **Voice AI applications** — phone-based voice agents, call centre bots, voice assistants, real-time transcription pipelines, and conversational AI systems. These systems have a fundamentally different requirement from generic noise cancellation: they must isolate exactly one speaker from a live audio stream, in real time, under production conditions — from callers phoning in from traffic, kitchens, crowded restaurants, and places full of background talkers.

Every architectural choice — the training data mix, the auxiliary separation objective, the 16 kHz native sample rate, the fully causal design — exists to serve this use case.

### At a production glance

| | |
|---|---|
| Model size | **8 MB** |
| Runs on | **CPU only — no GPU required** |
| Processing latency | **< 1 ms per 10 ms of audio** |
| Algorithmic latency | ~20 ms (fully causal, zero lookahead) |
| Training data | **10,000+ hours** of mixed speech, noise, and competing speakers |
| Sample rate | 16 kHz (telephony-native: G.711, WebRTC, SIP) |

The model processes each 10 ms audio frame in under 1 millisecond on a standard CPU core — meaning it adds no perceptible delay to live calls and leaves ample headroom for the rest of your pipeline.

---

## The Problem No Open-Source Model Solves

Voice agents for phone calls are deployed in the most uncontrolled acoustic environments imaginable: callers phoning in from moving cars and busy traffic, kitchens with running appliances, crowded restaurants, street corners, and homes where family members are talking in the background.

**The background speaker problem is the defining audio challenge of Voice AI** — and it is unsolved in open source.

Every major open-source speech enhancement model (DeepFilterNet3, RNNoise, SEGAN, MetricGAN+, and all DNS-Challenge entrants) is trained on **stationary noise** — fans, traffic, keyboard clicks, air conditioning. None of them treat a competing human voice as a first-class problem. When the interference is another person speaking, these models face a challenge their training never prepared them for: **speech looks like speech** in every feature they have learned. The result is one of two failures:

- The competing speaker leaks through and gets transcribed as part of the conversation — breaking downstream NLP, LLMs, and intent recognition entirely.
- The model suppresses both speakers indiscriminately, degrading the primary speaker's intelligibility.

**There is no open-source model that explicitly trains for background speaker suppression.** This gap exists because the DNS and CHiME benchmarks that drive open-source development measure noise suppression, not speaker isolation. Models optimized for those benchmarks are not optimized for Voice AI.

DeepFilterNet-SE is built to close that gap.

**A call with a noisy coffee machine is annoying. A call where a colleague in the background is being transcribed as part of the conversation breaks your entire Voice AI pipeline.**

---

## What DeepFilterNet-SE Does Differently

DeepFilterNet-SE is built on [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) — a strong, causal, real-time speech enhancement architecture — and extends it with one targeted change: **teaching the encoder to distinguish speakers, not just speech from noise.**

### The core insight

A noise cancellation encoder that has never been trained on competing speech has no reason to develop speaker-discriminative features. It learns a speech/noise boundary, not a this-speaker/that-speaker boundary. To fix this, you have to make the model care about speakers during training — not just at inference time.

### How we did it

**1. Training data that reflects the real problem**

60% of training samples include a competing human speaker drawn from a speaker-disjoint pool, mixed at 12–24 dB SIR below the primary. The model cannot pass training without learning to suppress speech that sounds like speech.

**2. Auxiliary Separation Head**

A lightweight `Linear(256→32) + Sigmoid` head is attached to the encoder bottleneck and trained with an L1 loss to predict an ERB-domain mask for the background speaker. This is an auxiliary training objective only — it does not add inference latency or complexity. Its sole purpose is to force the encoder representations to carry speaker-discriminative information that the main enhancement path can then exploit.

**3. Joint optimization**

The separation loss (weight 0.1) is combined with a multi-resolution spectral loss across 4 FFT scales (256/512/1024/2048). The separation objective regularizes the encoder without overriding the primary enhancement task.

At inference time, the Separation Head is discarded. Only the enhanced output path remains.

---

## Model at a Glance

| Parameter | Value |
|---|---|
| Model size | **8 MB** |
| CPU real-time | **Yes — no GPU required** |
| Processing latency | **< 1 ms per 10 ms frame** |
| Algorithmic latency | ~20 ms |
| Lookahead | 0 (fully causal) |
| Sample rate | 16,000 Hz |
| Frame size / hop | 320 / 160 samples (10 ms) |
| ERB bands | 32 |
| DF bins | 64 (order-5 filter) |
| Encoder dim | 256 |
| Total parameters | ~1.8M |
| Training data | 10,000+ hours |

16 kHz is the native sample rate for telephony (G.711, WebRTC, SIP) and for most speech models (Whisper, wav2vec2, etc.). Targeting 16 kHz directly avoids the artifacts introduced by resampling from wideband models.

The sub-millisecond per-frame processing time means the model runs in real time on a single CPU core with no added delay — critical for live voice agent deployments where any perceptible lag breaks the conversation.

---

## Architecture

```
Input Waveform [B, 1, T]
        |
        v
  STFT (FFT=320, Hop=160)
        |
   _____|_______________
   |                   |
   v                   v
ERB features        DF features
[B, 1, T, 32]      [B, 2, T, 64]
   |                   |
   '-------+------------'
           |
           v
        ENCODER
   (SqueezedGRU, 256-dim)
           |
   ________|____________________________
   |               |                   |
   v               v                   v
ERB DECODER     DF DECODER     SEPARATION HEAD *
(ConvTranspose  (3-layer GRU   (Linear + Sigmoid
 + skip conns)   + DF filter)   ERB-domain mask)
   |               |
   v               v
ERB gain mask   Complex filter
   |               |
   '-------+--------'
           |
           v
    Enhanced Spectrum
           |
           v
         ISTFT
           |
           v
   Enhanced Waveform [B, 1, T]
```

`*` Separation Head is active during training only. It is not used at inference time.

See [docs/architecture.md](docs/architecture.md) for full component details and [docs/architecture.png](docs/architecture.png) for a visual diagram.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/deepfilternet-se.git
cd deepfilternet-se
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

**Dependencies:** Python 3.9+, PyTorch 2.0+, torchaudio 2.0+, numpy, soundfile, scipy, h5py

---

## Quick Start: Inference

```python
import torch
import soundfile as sf
from model.dfnet_se import DfNetSE, get_config

config = get_config()
model = DfNetSE(config)
checkpoint = torch.load("checkpoints/model_best.ckpt", map_location="cpu")
model.model.load_state_dict(checkpoint)
model.eval()

audio, sr = sf.read("noisy_speech.wav")
assert sr == 16000, "Input must be 16 kHz"

wav = torch.tensor(audio).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
with torch.no_grad():
    enhanced = model(wav)  # [1, 1, T]

sf.write("enhanced.wav", enhanced.squeeze().numpy(), 16000)
```

### Batch Inference

```bash
python scripts/infer_dfnet_batch.py \
    --checkpoint checkpoints/model_best.ckpt \
    --input-dir /path/to/noisy_wavs \
    --output-dir /path/to/enhanced_wavs \
    --sr 16000
```

---

## Quick Start: Production Deployment

For production use without PyTorch, integrate the prebuilt **Weya NC Standalone** library directly into your C, C++, or Python application. See [deployment/README.md](deployment/README.md) for the full guide.

```python
# Python — no PyTorch required
import ctypes, platform, numpy as np

# Platform-aware library loading
lib_name = {"Darwin": "libweya_nc.dylib", "Windows": "weya_nc.dll"}.get(
    platform.system(), "libweya_nc.so"
)
lib = ctypes.CDLL(f"deployment/lib/{lib_name}")
# ... configure argtypes, then:
model = lib.weya_nc_model_load_from_path(b"path/to/model.tar.gz")
session = lib.weya_nc_session_create(model, 16000, ctypes.c_float(100.0))
frame_len = int(lib.weya_nc_get_frame_length(session))
lib.weya_nc_process_frame(session, input_ptr, output_ptr)
```

Prebuilt binaries are available for Linux, macOS (Apple Silicon), and Windows. Download them along with the ONNX model bundle from the [GitHub Releases](https://github.com/YOUR_USERNAME/deepfilternet-se/releases) page.

### Real-Time Examples

Test denoising live with your microphone or integrate into a streaming audio pipeline:

```bash
# Microphone demo — hear denoising in real time
pip install sounddevice
python3 deployment/examples/python/denoise_mic.py \
    --lib deployment/lib/libweya_nc.so \
    --model deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz

# WebSocket streaming server
pip install websockets
python3 deployment/examples/python/denoise_stream.py \
    --lib deployment/lib/libweya_nc.so \
    --model deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz
```

See [deployment/README.md](deployment/README.md) for protocol details and full usage instructions.

---

## Training

### Prerequisites

1. Prepare your training data as HDF5 shards — see [data/prepare_hdf5.md](data/prepare_hdf5.md)
2. Create a `dataset.cfg` JSON pointing to your HDF5 files

### Run Training

```bash
python training/train.py \
    --data-config data/hdf5_full/dataset.cfg \
    --data-dir data/hdf5_full \
    --base-dir runs/experiment_1 \
    --config-ini configs/default.ini
```

Debug mode (CPU, 5 steps):

```bash
python training/train.py \
    --data-config data/hdf5_full/dataset.cfg \
    --data-dir data/hdf5_full \
    --base-dir runs/debug \
    --device cpu --debug --max-steps 5
```

### Training Configuration

The released checkpoint was trained with the exact configuration in `configs/default.ini`. Key settings:

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4 |
| LR schedule | Cosine with 3-epoch warmup |
| Epochs | 100 (early stop patience=25) |
| Batch size | 16 |
| Loss | MultiResSpecLoss (4 scales) + LocalSNRLoss + SeparationLoss |
| Background speaker prob. | 60% of samples |
| Background SIR range | 12–24 dB |

---

## Data Preparation

Training uses the [libdfdata](https://github.com/Rikorose/DeepFilterNet/tree/main/libDF) HDF5-backed dataloader. Data is organized in four categories:

| Category | Role | Description |
|---|---|---|
| `train_speech` | Clean target | Primary speaker recordings |
| `background_speech` | Interferer pool | Background human voices (speaker-disjoint from train_speech) |
| `noise` | Noise augmentation | Environmental / background noise |
| `rir` | Reverberation | Room impulse responses |

Full instructions: [data/prepare_hdf5.md](data/prepare_hdf5.md)

---

## Datasets

The model was trained on standard publicly available datasets totalling **over 10,000 hours of mixed audio** — clean speech, competing speakers, environmental noise, and room impulse responses. See [DATASETS.md](DATASETS.md) for the full list with links and licensing information.

**Summary:**
- **Speech**: LibriSpeech, VCTK, Common Voice (English)
- **Background speech**: LibriSpeech / VCTK (speaker-disjoint splits)
- **Noise**: DNS Challenge, FreeSound, ESC-50
- **RIRs**: MIT IR Survey, OpenAIR, BUT ReverbDB

---

## Project Structure

```
deepfilternet-se/
├── model/
│   ├── dfnet_se.py          # Full model: DfNetSE, DfNet, Encoder, ErbDecoder,
│   │                        #   DfDecoder, SeparationHead
│   └── __init__.py
├── training/
│   ├── train.py             # Training loop (AdamW + cosine LR + early stopping)
│   ├── loss.py              # Loss functions: MaskLoss, SpectralLoss,
│   │                        #   MultiResSpecLoss, SdrLoss, SeparationLoss, Loss
│   ├── config.py            # Dataclass config (DFConfig, ModelConfig, etc.)
│   ├── lr.py                # Cosine LR scheduler
│   └── __init__.py
├── scripts/
│   ├── infer_dfnet_batch.py         # Batch inference
│   ├── compute_objective_metrics.py # SI-SDR, STOI, ESTOI evaluation
│   └── create_simple_demo_dataset.py # Demo dataset creation
├── data/
│   ├── prepare_hdf5.md      # Data preparation guide
│   ├── scripts/
│   │   ├── prepare_hdf5.sh          # Convert JSON manifests → HDF5 shards
│   │   ├── build_rust_data_stack.sh # Build hdf5_prep binary + libdf extensions
│   │   └── validate_hdf5_contract.sh # Validate HDF5 files meet sr/dtype contract
│   └── manifests/
│       ├── README.md                # Manifest JSON schema reference
│       ├── example_train.json       # Example speech manifest
│       ├── example_noise.json       # Example noise manifest
│       ├── example_rir.json         # Example RIR manifest
│       └── example_background.json  # Example background speech manifest
├── deployment/
│   ├── README.md            # Integration guide for C/C++/Python (no PyTorch)
│   ├── include/
│   │   └── weya_nc.h        # C API header (10 functions)
│   └── examples/
│       ├── c/
│       │   └── denoise_frame_demo.c    # Canonical C usage example
│       └── python/
│           ├── weya_nc.py                # Reusable Python wrapper for C API
│           ├── denoise_ctypes.py         # File-based WAV denoising example
│           ├── denoise_mic.py            # Real-time microphone demo
│           └── denoise_stream.py         # WebSocket streaming pipeline
├── docs/
│   ├── architecture.md      # Detailed architecture description
│   └── architecture.png     # Architecture diagram
├── configs/
│   └── default.ini          # Training config used for released checkpoint
├── DATASETS.md              # Dataset sources and licensing
├── GITHUB_RELEASE_NOTES.md  # v1.0.0 release notes
├── requirements.txt
├── setup.py
└── LICENSE                  # Apache 2.0
```

---

## Acknowledgements

This project is built on [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) by Hendrik Schröter, Tobias Rosenkranz, Alberto N. Escalante-B., and Andreas Maier. The core architecture, ERB filterbank implementation, SqueezedGRU module, and loss functions closely follow the DF3 design. The `libdfdata` Rust library for HDF5-backed data loading is from the same project.

---

## Citation

If you use this model or code, please cite the original DeepFilterNet paper:

```bibtex
@inproceedings{schroter2023deepfilternet3,
  title     = {DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement},
  author    = {Schröter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N and Maier, Andreas},
  booktitle = {INTERSPEECH},
  year      = {2023}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
