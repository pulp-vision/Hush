#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/hdf5}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import h5py
PY
then
  echo "Python runtime check failed for '$PYTHON_BIN'."
  echo "Set PYTHON_BIN to an interpreter with h5py installed."
  exit 1
fi

"$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import json
import os
import sys
from glob import glob

import h5py

data_dir = sys.argv[1]

def check_file(path: str, expected_group: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if expected_group not in f:
            raise ValueError(f"{path}: missing group '{expected_group}'")
        sr = int(f.attrs.get("sr", -1))
        codec = str(f.attrs.get("codec", ""))
        dtype = str(f.attrs.get("dtype", ""))
        if sr != 16000:
            raise ValueError(f"{path}: expected sr=16000, got {sr}")
        if "pcm" not in codec:
            raise ValueError(f"{path}: expected codec=pcm, got {codec}")
        if "float32" not in dtype:
            raise ValueError(f"{path}: expected dtype=float32, got {dtype}")
        print(f"OK {os.path.basename(path)} group={expected_group} sr={sr}")

def collect(pattern: str):
    return sorted(glob(os.path.join(data_dir, pattern)))

train_paths = collect("train_speech*.hdf5")
noise_paths = collect("noise*.hdf5")
val_paths = collect("val_speech*.hdf5")
rir_paths = collect("rir*.hdf5")
bg_paths = collect("background_speech*.hdf5")

if not train_paths:
    raise ValueError("No train_speech*.hdf5 files found")
if not noise_paths:
    raise ValueError("No noise*.hdf5 files found")

for p in train_paths:
    check_file(p, "speech")
for p in noise_paths:
    check_file(p, "noise")
for p in val_paths:
    check_file(p, "speech")
for p in rir_paths:
    check_file(p, "rir")
for p in bg_paths:
    check_file(p, "speech")

cfg_path = os.path.join(data_dir, "dataset.cfg")
if not os.path.isfile(cfg_path):
    raise FileNotFoundError(cfg_path)

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

for split in ("train", "valid", "test"):
    if split not in cfg:
        raise ValueError(f"dataset.cfg missing split '{split}'")

train_entries = [x[0] for x in cfg["train"]]
valid_entries = [x[0] for x in cfg["valid"]]
test_entries = [x[0] for x in cfg["test"]]

def ensure_all_present(entries, names, desc):
    missing = sorted(set(names) - set(entries))
    if missing:
        raise ValueError(f"{desc} missing entries: {missing}")

train_names = [os.path.basename(p) for p in train_paths]
noise_names = [os.path.basename(p) for p in noise_paths]
val_names = [os.path.basename(p) for p in val_paths]
rir_names = [os.path.basename(p) for p in rir_paths]
bg_names = [os.path.basename(p) for p in bg_paths]

ensure_all_present(train_entries, train_names, "dataset.cfg train split (train_speech)")
ensure_all_present(train_entries, noise_names, "dataset.cfg train split (noise)")
if rir_names:
    ensure_all_present(train_entries, rir_names, "dataset.cfg train split (rir)")
if bg_names:
    ensure_all_present(train_entries, bg_names, "dataset.cfg train split (background)")

if val_names:
    ensure_all_present(valid_entries, val_names, "dataset.cfg valid split (val_speech)")
    ensure_all_present(test_entries, val_names, "dataset.cfg test split (val_speech)")

print("dataset.cfg split contract looks valid")
PY
