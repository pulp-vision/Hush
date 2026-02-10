# DeepFilterNet-SE

**Primary Speaker Extraction + Denoising** at 16 kHz, CPU-friendly, ONNX-exportable.

## Overview

DeepFilterNet-SE extends [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) to perform joint denoising and background speaker suppression. The model is trained on synthetic mixtures where the target is the clean primary speaker.

## Features

- **16 kHz phone-band** optimized for telephony/VoIP
- **Real-time capable** on single CPU thread (RTF < 1.0)
- **ONNX exportable** with dynamic time axis
- **Streaming-friendly** frame-based processing

## Quick Start

### Installation

```bash
pip install -e ".[train]"
```

### Prepare Data

```bash
python scripts/prepare_manifests.py --data-root /path/to/datasets --output-dir data/manifests
```

### Train

```bash
# Debug run (3 epochs, small subset)
python training/train.py --manifest-dir data/manifests --debug

# Full training
python training/train.py --manifest-dir data/manifests --epochs 100
```

### Export to ONNX

```bash
python export/onnx_export.py --checkpoint checkpoints/best.pt --output-dir exports --benchmark
```

### Evaluate

```bash
python evaluation/evaluate.py --checkpoint checkpoints/best.pt --test-manifest data/manifests/test.json
```

## Architecture

Based on DeepFilterNet3 with adaptations for 16 kHz:

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16,000 Hz |
| FFT Size | 512 |
| Hop Size | 128 |
| ERB Bands | 32 |
| DF Bins | 64 |
| DF Order | 5 |

## License

MIT / Apache 2.0 (following DeepFilterNet)
