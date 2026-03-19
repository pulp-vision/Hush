# Data Preparation Guide

DeepFilterNet-SE uses the [libdfdata](https://github.com/Rikorose/DeepFilterNet/tree/main/libDF) Rust-based dataloader for efficient HDF5-backed training. This guide walks through the complete pipeline using the scripts in `data/scripts/`.

## Overview

Training data is organized into four categories, each packed into one or more HDF5 shards:

| Category | HDF5 filename prefix | HDF5 group | Sampling weight | Role |
|---|---|---|---|---|
| `train_speech` | `train_speech*.hdf5` | `speech` | 10 | Primary speaker (clean target) |
| `background_speech` | `background_speech*.hdf5` | `speech` | 1 | Background/interferer speakers |
| `noise` | `noise*.hdf5` | `noise` | 10 | Environmental noise |
| `rir` | `rir*.hdf5` | `rir` | 1 | Room impulse responses |

Higher sampling weights mean those shards are drawn more frequently during batch construction.

---

## Step 0: Build the Rust Data Stack

The `hdf5_prep` binary and the `libdfdata`/`libdf` Python extensions must be compiled from the DeepFilterNet Rust workspace before packing any data.

```bash
# Point RUST_DIR to your local DeepFilterNet rust/ directory
RUST_DIR=/path/to/DeepFilterNet/rust \
PYTHON_BIN=$(which python3) \
  bash data/scripts/build_rust_data_stack.sh
```

`build_rust_data_stack.sh` does three things:
1. Compiles `hdf5_prep` via `cargo build --release -p hdf5_prep`
2. Installs the `libdf` Python extension (`pyDF/`)
3. Installs the `libdfdata` Python extension (`pyDF-data/`)

It skips steps 2 and 3 if those packages are already importable (set `FORCE_REINSTALL=1` to rebuild).

**Prerequisites:**
- Rust toolchain (`rustup`): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Python with `maturin` and `h5py`: `pip install maturin h5py`

---

## Step 1: Prepare JSON Manifests

Each audio category needs a JSON manifest listing the files to pack. See [data/manifests/README.md](manifests/README.md) for the full schema, and the `example_*.json` files for reference.

### Required manifests

| File | Content |
|---|---|
| `train.json` | Clean speech for training (e.g. LibriSpeech train-clean-100/360, VCTK) |
| `val.json` | Validation speech (speaker-disjoint from train) |
| `noise.json` | Environmental noise (e.g. DNS Challenge, FreeSound, ESC-50) |
| `rir.json` | Room impulse responses (e.g. MIT IR Survey, OpenAIR, BUT ReverbDB) |
| `background.json` | Background speaker pool (speaker-disjoint from train) — optional |

All audio must be pre-resampled to **16 kHz mono**:

```bash
for f in /path/to/raw/*.wav; do
    sox "$f" -r 16000 -c 1 /path/to/16k/$(basename "$f")
done
```

See [DATASETS.md](../DATASETS.md) for the full list of datasets used in the released checkpoint.

---

## Step 2: Run `prepare_hdf5.sh`

```bash
bash data/scripts/prepare_hdf5.sh \
    /path/to/manifests \
    /path/to/datasets \
    /path/to/output/hdf5
```

**Positional arguments:**

| Position | Variable | Default | Description |
|---|---|---|---|
| 1 | `MANIFEST_DIR` | `data/manifests` (from repo root) | Directory containing `train.json`, `val.json`, etc. |
| 2 | `DATA_ROOT` | parent of repo root | Root directory that manifest paths are relative to |
| 3 | `OUT_DIR` | `data/hdf5` (from repo root) | Output directory for HDF5 files |
| 4 | `SR` | `16000` | Target sample rate |

**Environment variable overrides:**

| Variable | Default | Description |
|---|---|---|
| `SKIP_EXISTING` | `1` | Skip if output exists and passes sanity check |
| `PYTHON_BIN` | `/usr/bin/python3` | Python interpreter with h5py |
| `MAX_WORKERS` | `nproc` | Maximum parallel worker processes |
| `TRAIN_SHARDS` | `MAX_WORKERS` | Number of train shards |
| `TRAIN_WORKERS` | `MAX_WORKERS` | Parallel workers for train packing |
| `VAL_SHARDS` | `1` | Number of validation shards |
| `NOISE_SHARDS` | `MAX_WORKERS` | Number of noise shards |
| `RIR_SHARDS` | `MAX_WORKERS` | Number of RIR shards |
| `BG_SHARDS` | `MAX_WORKERS * 4` | Number of background speech shards |
| `TRAIN_LIMIT` | _(none)_ | Cap on number of train samples (for testing) |
| `NOISE_LIMIT` | _(none)_ | Cap on number of noise samples |
| `MIN_FREE_GB` | `80` | Abort if less than this much disk space available |
| `CLEANUP_MODE` | `aggressive` | Remove `.tmp`/`.part` files on start and finish |
| `TRAIN_SPEECH_FACTOR` | `10` | Sampling weight for train speech in dataset.cfg |
| `NOISE_FACTOR` | `10` | Sampling weight for noise in dataset.cfg |
| `RIR_FACTOR` | `1` | Sampling weight for RIRs in dataset.cfg |
| `BACKGROUND_SPEECH_FACTOR` | `1` | Sampling weight for background speech in dataset.cfg |

**Example with overrides:**

```bash
TRAIN_SHARDS=24 NOISE_SHARDS=24 RIR_SHARDS=8 BG_SHARDS=32 \
MIN_FREE_GB=200 \
PYTHON_BIN=/home/user/.venv/bin/python \
  bash data/scripts/prepare_hdf5.sh \
    /data/manifests \
    /data/datasets \
    /data/hdf5_full
```

### Output: `dataset.cfg`

After packing all shards, the script writes `$OUT_DIR/dataset.cfg` — a JSON file that maps HDF5 shard filenames to sampling weights for each split:

```json
{
  "train": [
    ["train_speech_000.hdf5", 10],
    ["train_speech_001.hdf5", 10],
    ["background_speech_000.hdf5", 1],
    ["noise_000.hdf5", 10],
    ["rir_000.hdf5", 1]
  ],
  "valid": [
    ["val_speech.hdf5", 10],
    ["noise_000.hdf5", 10],
    ["rir_000.hdf5", 1]
  ],
  "test": [
    ["val_speech.hdf5", 10],
    ["noise_000.hdf5", 10],
    ["rir_000.hdf5", 1]
  ]
}
```

Pass this file to `training/train.py` via `--data-config`.

---

## Step 3: Validate with `validate_hdf5_contract.sh`

```bash
bash data/scripts/validate_hdf5_contract.sh /path/to/output/hdf5
```

This checks every HDF5 file against the contract expected by the dataloader:
- Sample rate is 16000 Hz
- Codec is PCM
- Dtype is float32
- Required HDF5 groups are present (`speech`, `noise`, `rir`)
- `dataset.cfg` contains all expected splits and shard entries

A passing run prints `OK` for each file and `dataset.cfg split contract looks valid` at the end.

---

## Step 4: Smoke-Test with the Demo Script

Run a quick end-to-end check using the demo dataset creator (creates audible mixed samples and optionally runs model inference):

```bash
python scripts/create_simple_demo_dataset.py \
    --data-dir /path/to/output/hdf5 \
    --out-dir /tmp/demo_samples
```

Or smoke-test training itself (CPU, 5 steps):

```bash
python training/train.py \
    --data-config /path/to/output/hdf5/dataset.cfg \
    --data-dir /path/to/output/hdf5 \
    --base-dir runs/smoke_test \
    --device cpu --debug --max-steps 5
```

---

## Notes on Background Speech

The `background_speech` shards must contain **speaker-disjoint** speakers from `train_speech`. At training time, the dataloader mixes a background speaker into 60% of training samples at 12–24 dB SIR below the primary speaker. This is what trains the model's background speaker suppression capability.

## Notes on RIRs

Room impulse responses are applied to 10% of samples (`p_reverb = 0.1`). Any standard RIR dataset works; see [DATASETS.md](../DATASETS.md) for recommendations.
