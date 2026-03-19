# Release v1.0.0 — DeepFilterNet-SE: Real-Time Speech Enhancement for Voice AI

This is the initial open-source release of **DeepFilterNet-SE** — the first open-source speech enhancement model built specifically for **Voice AI** applications: phone-based voice agents, call centre bots, real-time transcription, and conversational AI.

It is an **8 MB model** that runs fully on CPU in real time, processes each 10 ms audio frame in under 1 millisecond, and was trained on over **10,000 hours** of mixed audio — including competing human speakers — to handle the real-world conditions voice agents face: callers phoning from traffic, kitchens, crowded spaces, and environments where other people are talking in the background.

Built on [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet), it extends the architecture with a novel auxiliary separation head that explicitly teaches the model to suppress background human voices — the one failure mode that standard open-source noise cancellation models do not solve.

---

## What's in This Release

- **Complete source code**: model architecture, training loop, loss functions, LR scheduler, inference scripts
- **Pretrained checkpoint**: `model_best.ckpt` — best validation loss over 100 training epochs
- **Training configuration**: `configs/default.ini` — exact config used for the released model
- **Data preparation guide**: `data/prepare_hdf5.md` — how to build your own HDF5 dataset
- **Dataset documentation**: `DATASETS.md` — all datasets used, with links
- **Architecture documentation**: `docs/architecture.md` and `docs/architecture.png`
- **Real-time deployment examples**: Microphone demo (`denoise_mic.py`), WebSocket streaming server (`denoise_stream.py`), and reusable Python wrapper (`weya_nc.py`)

---

## Key Features

- **Built for Voice AI** — designed specifically for phone-based voice agents and real-time transcription pipelines, not generic noise benchmarks
- **8 MB model size** — lightweight enough to embed in any production stack, no large model infrastructure required
- **Runs fully on CPU in real time** — no GPU needed; processes audio on a single CPU core without any perceptible delay
- **Under 1 ms per 10 ms of audio** — each frame is processed in under 1 millisecond, keeping end-to-end latency imperceptible on live calls
- **10,000+ hours of training data** — trained on a diverse mix of speech, competing speakers, environmental noise, and room acoustics
- **16 kHz optimized** — telephony-native sample rate (G.711, WebRTC, SIP), compatible with Whisper, wav2vec2, and all major speech models
- **Novel Separation Head** — lightweight Linear(256→32)+Sigmoid auxiliary head that learns to predict ERB-domain background speaker masks, forcing the encoder to develop speaker-discriminative representations
- **Background speaker suppression** — trained with 60% of samples containing a competing human voice at 12–24 dB SIR, directly addressing the failure mode that breaks downstream NLP in Voice AI
- **Multi-resolution spectral training** — loss computed across 4 FFT scales (256, 512, 1024, 2048) for robust frequency coverage
- **Fully self-contained** — no dependency on the original DeepFilterNet package for inference

---

## Architecture Highlights

```
Input → STFT → [ERB features, DF features]
             → Encoder (depthwise-sep Conv2d + SqueezedGRU, 256-dim)
             → ERB Decoder (ConvTranspose + skip connections → gain mask)
             → DF Decoder (3-layer GRU → complex filter coefficients)
             → Separation Head [NEW] (Linear+Sigmoid → background speaker mask)
             → Enhanced Spectrum → ISTFT → Output
```

| Component | Details |
|---|---|
| Model size | **8 MB** |
| CPU real-time | **Yes — no GPU required** |
| Processing latency | **< 1 ms per 10 ms frame** |
| Training data | **10,000+ hours** |
| Sample rate | 16,000 Hz |
| FFT / Hop | 320 / 160 (10 ms frames) |
| ERB bands | 32 |
| DF bins | 64 (order-5 filter) |
| Embedding dim | 256 |
| Parameters | ~1.8M |
| Lookahead | 0 (fully causal) |

---

## Background Speaker Removal Strategy

Voice AI products face a problem standard speech enhancement models were never trained to solve: callers phoning from traffic, kitchens, and environments where other people are talking nearby. Standard models are trained on noise-only interference and fail when the "noise" is another human voice — they tend to either suppress the primary speaker or leave the background speaker intact, causing it to be transcribed as part of the conversation and breaking downstream NLP entirely.

This model addresses this with a three-part approach:

1. **Dataset augmentation**: 60% of training samples include a background speaker at SIR ∈ {12, 14, 16, 18, 20, 24} dB. This is drawn from a speaker-disjoint pool to prevent memorization.

2. **Auxiliary Separation Head**: a tiny Linear head attached to the encoder bottleneck, supervised with an L1 loss to predict an ERB gain mask for the background speaker signal. This regularizes the encoder to develop features that distinguish speakers, not just speech from noise.

3. **Joint loss**: the separation loss (factor=0.1) is combined with multi-resolution spectral losses so the separation objective never dominates the primary enhancement task.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 (cosine decay to 1e-6) |
| LR warmup | 3 epochs (1e-4 → 5e-4) |
| Weight decay | 0.05 |
| Betas | (0.9, 0.999), AMSGrad |
| Batch size | 16 |
| Max sample length | 5 seconds |
| Epochs | 100 |
| Early stopping | patience=25 epochs |
| Gradient clip | 1.0 |
| Loss | MRSL (×250 mag + ×250 complex, γ=0.3, FFTs=[256,512,1024,2048]) + LocalSNR (×0.001) + SeparationLoss (×0.1) |

---

## Datasets Used

Training data (16 kHz, HDF5 format):

Total training data: **10,000+ hours** of mixed audio across all four categories.

| Category | Source | Shards | Sampling Weight |
|---|---|---|---|
| Primary speech | LibriSpeech, VCTK, Common Voice | 24 | ×10 |
| Background speech | LibriSpeech/VCTK (speaker-disjoint) | 24 | ×1 |
| Noise | DNS Challenge, FreeSound, ESC-50 | 24 | ×10 |
| Room impulse responses | MIT IR Survey, OpenAIR, BUT ReverbDB | 24 | ×1 |

See [DATASETS.md](DATASETS.md) for full details including URLs and licensing.

---

## How to Use

```python
import torch
from model.dfnet_se import DfNetSE, get_config

# Load model
config = get_config()
model = DfNetSE(config)

checkpoint = torch.load("checkpoints/model_best.ckpt", map_location="cpu")
model.model.load_state_dict(checkpoint)
model.eval()

# Run inference
with torch.no_grad():
    noisy_wav = torch.randn(1, 1, 16000)  # [B, 1, T] at 16 kHz
    enhanced = model(noisy_wav)           # [B, 1, T]
```

---

## Known Limitations

- **16 kHz only** — the model was trained and evaluated at 16 kHz. Using it at other sample rates requires resampling.
- **Separation head is auxiliary** — the background speaker mask from the separation head is an ERB-domain soft mask, not a full waveform separator. It regularizes training but is not intended for use as a standalone source separation output.
- **Background speaker at moderate SIR** — the model was trained with background speakers at 12-24 dB SIR. Very loud competing speakers (low SIR) may not be fully suppressed.

---

## Acknowledgements

This work is built on the excellent [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) by Hendrik Schröter et al. The core architecture, ERB filterbank, SqueezedGRU, and loss formulation follow the DF3 design closely. The training dataloader uses the `libdfdata` Rust library from the same project.

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
