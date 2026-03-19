#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUST_DIR="$ROOT_DIR/rust"
HDF5_PREP_BIN="$RUST_DIR/target/release/hdf5_prep"

MANIFEST_DIR="${1:-$ROOT_DIR/../data/manifests}"
DATA_ROOT="${2:-$ROOT_DIR/..}"
OUT_DIR="${3:-$ROOT_DIR/data/hdf5}"
SR="${4:-16000}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
TRAIN_SPEECH_FACTOR="${TRAIN_SPEECH_FACTOR:-10}"
BACKGROUND_SPEECH_FACTOR="${BACKGROUND_SPEECH_FACTOR:-1}"
VALID_BACKGROUND_FACTOR="${VALID_BACKGROUND_FACTOR:-1}"
INCLUDE_BG_IN_VALID_TEST="${INCLUDE_BG_IN_VALID_TEST:-0}"
NOISE_FACTOR="${NOISE_FACTOR:-10}"
RIR_FACTOR="${RIR_FACTOR:-1}"
MAX_WORKERS="${MAX_WORKERS:-$(nproc)}"
TRAIN_SHARDS="${TRAIN_SHARDS:-$MAX_WORKERS}"
TRAIN_WORKERS="${TRAIN_WORKERS:-$MAX_WORKERS}"
VAL_SHARDS="${VAL_SHARDS:-1}"
VAL_WORKERS="${VAL_WORKERS:-1}"
NOISE_SHARDS="${NOISE_SHARDS:-$MAX_WORKERS}"
NOISE_WORKERS="${NOISE_WORKERS:-$MAX_WORKERS}"
RIR_SHARDS="${RIR_SHARDS:-$MAX_WORKERS}"
RIR_WORKERS="${RIR_WORKERS:-$MAX_WORKERS}"
BG_SHARDS="${BG_SHARDS:-$((MAX_WORKERS * 4))}"
BG_WORKERS="${BG_WORKERS:-$MAX_WORKERS}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
NOISE_LIMIT="${NOISE_LIMIT:-}"
RIR_LIMIT="${RIR_LIMIT:-}"
BG_LIMIT="${BG_LIMIT:-}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
CLEANUP_MODE="${CLEANUP_MODE:-aggressive}"
CLEANUP_TMP_SHARDS="${CLEANUP_TMP_SHARDS:-1}"

mkdir -p "$OUT_DIR"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import h5py
PY
then
  echo "Python runtime check failed for '$PYTHON_BIN'."
  echo "Set PYTHON_BIN to an interpreter with h5py installed."
  exit 1
fi

is_valid_existing_output() {
  local output="$1"
  local group="$2"
  local manifest="$3"
  local limit="${4:-}"
  "$PYTHON_BIN" - "$output" "$group" "$manifest" "$SR" "$limit" <<'PY'
import json
import sys

output, group, manifest, sr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
limit = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None

try:
    import h5py

    with h5py.File(output, "r") as f:
        if group not in f:
            print(f"invalid: missing group '{group}'")
            sys.exit(2)
        file_sr = int(f.attrs.get("sr", -1))
        if file_sr != sr:
            print(f"invalid: sr={file_sr} expected={sr}")
            sys.exit(2)
        count = len(f[group])
except Exception as exc:
    print(f"invalid: {type(exc).__name__}: {exc}")
    sys.exit(2)

expected = None
try:
    with open(manifest, "r", encoding="utf-8") as f:
        expected = len(json.load(f).get("samples", []))
except Exception:
    expected = None

if expected is not None and expected > 0 and limit is not None and limit > 0:
    expected = min(expected, limit)

if expected is None or expected <= 0:
    print(f"valid: count={count}")
    sys.exit(0)

min_ok = max(1, int(expected * 0.95))
if count < min_ok:
    print(f"invalid: count={count} expected~{expected} (<{min_ok})")
    sys.exit(2)

print(f"valid: count={count} expected~{expected}")
PY
}

check_free_space() {
  local path="$1"
  local min_gb="${2:-$MIN_FREE_GB}"
  local avail_kb
  avail_kb="$(df -Pk "$path" | awk 'NR==2 {print $4}')"
  local avail_gb=$((avail_kb / 1024 / 1024))
  echo "[disk] available=${avail_gb}GiB required>=${min_gb}GiB path=$path"
  if (( avail_gb < min_gb )); then
    echo "[disk] ERROR: insufficient free space"
    exit 1
  fi
}

cleanup_generated_artifacts() {
  local mode="$1"
  if [[ "$mode" != "aggressive" ]]; then
    return 0
  fi
  find "$OUT_DIR" -maxdepth 1 -type f -name "*.tmp" -delete
  find "$OUT_DIR" -maxdepth 1 -type f -name "*.part" -delete
  find "$OUT_DIR" -maxdepth 1 -type f -name "*.failed" -delete
  find "$OUT_DIR" -maxdepth 1 -type f -name "*.hdf5" -size 0c -delete
}

run_prep() {
  local manifest="$1"
  local output="$2"
  local group="$3"
  local limit="${4:-}"
  local prefix
  prefix="$(basename "$output")"

  if [[ ! -f "$manifest" ]]; then
    echo "[$prefix] Skipping missing manifest: $manifest"
    return
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$output" ]]; then
    local check_msg
    if check_msg="$(is_valid_existing_output "$output" "$group" "$manifest" "$limit")"; then
      echo "[$prefix] Skipping existing output: $output ($check_msg)"
      return
    fi
    echo "[$prefix] Existing output failed sanity check, rebuilding: $output"
    echo "[$prefix] Reason: $check_msg"
    rm -f "$output"
  fi
  if [[ "$SKIP_EXISTING" != "1" && -f "$output" ]]; then
    echo "[$prefix] Removing existing output before rebuild: $output"
    rm -f "$output"
  fi

  if [[ -n "$limit" ]]; then
    "$HDF5_PREP_BIN" \
      --manifest "$manifest" \
      --output "$output" \
      --group "$group" \
      --working-dir "$DATA_ROOT" \
      --sr "$SR" \
      --max-freq "$((SR / 2))" \
      --limit "$limit" \
      2>&1 | sed -u "s/^/[$prefix] /"
  else
    "$HDF5_PREP_BIN" \
      --manifest "$manifest" \
      --output "$output" \
      --group "$group" \
      --working-dir "$DATA_ROOT" \
      --sr "$SR" \
      --max-freq "$((SR / 2))" \
      2>&1 | sed -u "s/^/[$prefix] /"
  fi
}

split_manifest() {
  local manifest="$1"
  local out_dir="$2"
  local prefix="$3"
  local shards="$4"
  local limit="${5:-}"
  "$PYTHON_BIN" - "$manifest" "$out_dir" "$prefix" "$shards" "$limit" <<'PY'
import json
import os
import sys

manifest, out_dir, prefix, shards = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
limit = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
os.makedirs(out_dir, exist_ok=True)
with open(manifest, "r", encoding="utf-8") as f:
    data = json.load(f)
samples = data.get("samples", [])
if limit is not None and limit > 0:
    samples = samples[:limit]
for i in range(shards):
    part = samples[i::shards]
    path = os.path.join(out_dir, f"{prefix}_{i:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": part}, f)
print(f"Split {len(samples)} entries into {shards} shards at {out_dir} (limit={limit})")
PY
}

run_manifest_parallel() {
  local manifest="$1"
  local out_prefix="$2"
  local group="$3"
  local shard_dir="$4"
  local shards="$5"
  local workers="$6"
  local limit="${7:-}"
  local part_prefix="${out_prefix}_part"
  local running=0
  local failed=0

  if [[ ! -f "$manifest" ]]; then
    echo "[$out_prefix] Skipping missing manifest: $manifest"
    return 0
  fi

  # Remove stale shard files from previous runs with a different shard count.
  # This prevents old/corrupt outputs from being accidentally included by the
  # dataset.cfg glob stage.
  shopt -s nullglob
  local stale_removed=0
  local f base shard_txt shard_id
  for f in "$OUT_DIR/${out_prefix}_"*.hdf5; do
    base="$(basename "$f")"
    shard_txt="${base#${out_prefix}_}"
    shard_txt="${shard_txt%.hdf5}"
    if [[ "$shard_txt" =~ ^[0-9]{3}$ ]]; then
      shard_id=$((10#$shard_txt))
      if (( shards <= 1 || shard_id >= shards )); then
        rm -f "$f"
        stale_removed=$((stale_removed + 1))
      fi
    fi
  done
  if (( stale_removed > 0 )); then
    echo "[$out_prefix] Removed $stale_removed stale shard outputs."
  fi

  if (( shards <= 1 )); then
    run_prep "$manifest" "$OUT_DIR/${out_prefix}.hdf5" "$group" "$limit"
    return 0
  fi

  rm -rf "$shard_dir"
  split_manifest "$manifest" "$shard_dir" "$part_prefix" "$shards" "$limit"

  shopt -s nullglob
  local manifests=("$shard_dir"/"${part_prefix}"_*.json)
  if [[ ${#manifests[@]} -eq 0 ]]; then
    echo "[$out_prefix] No shard manifests were created."
    return 1
  fi

  echo "Preparing ${out_prefix} with ${#manifests[@]} shards and ${workers} workers"
  for mf in "${manifests[@]}"; do
    local shard_id
    shard_id="$(basename "$mf" .json | sed "s/${part_prefix}_//")"
    local output="$OUT_DIR/${out_prefix}_${shard_id}.hdf5"

    (
      set +e
      run_prep "$mf" "$output" "$group"
      ec=$?
      if (( ec != 0 )); then
        echo "[${out_prefix}_${shard_id}.hdf5] ERROR: shard prep failed with exit code ${ec}"
        rm -f "$output"
      fi
      exit $ec
    ) &
    running=$((running + 1))
    if (( running >= workers )); then
      if ! wait -n; then
        failed=1
      fi
      running=$((running - 1))
    fi
  done
  while (( running > 0 )); do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done

  if (( failed != 0 )); then
    echo "[${out_prefix}] ERROR: One or more shard jobs failed."
    return 1
  fi

  if [[ "$CLEANUP_TMP_SHARDS" == "1" ]]; then
    rm -rf "$shard_dir"
  fi
}

echo "Preparing HDF5 datasets into: $OUT_DIR"
echo "Options: SKIP_EXISTING=$SKIP_EXISTING MAX_WORKERS=$MAX_WORKERS"
echo "Shards/workers:"
echo "  train: $TRAIN_SHARDS/$TRAIN_WORKERS"
echo "  val: $VAL_SHARDS/$VAL_WORKERS"
echo "  noise: $NOISE_SHARDS/$NOISE_WORKERS"
echo "  rir: $RIR_SHARDS/$RIR_WORKERS"
echo "  background: $BG_SHARDS/$BG_WORKERS"
echo "Sampling factors: train_speech=$TRAIN_SPEECH_FACTOR background=$BACKGROUND_SPEECH_FACTOR valid_background=$VALID_BACKGROUND_FACTOR noise=$NOISE_FACTOR rir=$RIR_FACTOR"
echo "Validation background: INCLUDE_BG_IN_VALID_TEST=$INCLUDE_BG_IN_VALID_TEST"
echo "Limits: train=$TRAIN_LIMIT val=$VAL_LIMIT noise=$NOISE_LIMIT rir=$RIR_LIMIT bg=$BG_LIMIT"
echo "Disk policy: MIN_FREE_GB=$MIN_FREE_GB CLEANUP_MODE=$CLEANUP_MODE CLEANUP_TMP_SHARDS=$CLEANUP_TMP_SHARDS"
check_free_space "$OUT_DIR" "$MIN_FREE_GB"
cleanup_generated_artifacts "$CLEANUP_MODE"
if [[ ! -x "$HDF5_PREP_BIN" ]]; then
  echo "Building hdf5_prep binary once for fast execution"
  cargo build --release -p hdf5_prep --manifest-path "$RUST_DIR/Cargo.toml"
fi
check_free_space "$OUT_DIR" "$MIN_FREE_GB"
run_manifest_parallel "$MANIFEST_DIR/train.json" "train_speech" "speech" "$OUT_DIR/.manifest_shards/train_speech" "$TRAIN_SHARDS" "$TRAIN_WORKERS" "$TRAIN_LIMIT"
check_free_space "$OUT_DIR" "$MIN_FREE_GB"
run_manifest_parallel "$MANIFEST_DIR/val.json" "val_speech" "speech" "$OUT_DIR/.manifest_shards/val_speech" "$VAL_SHARDS" "$VAL_WORKERS" "$VAL_LIMIT"
check_free_space "$OUT_DIR" "$MIN_FREE_GB"
run_manifest_parallel "$MANIFEST_DIR/noise.json" "noise" "noise" "$OUT_DIR/.manifest_shards/noise" "$NOISE_SHARDS" "$NOISE_WORKERS" "$NOISE_LIMIT"
RIR_MANIFEST="$MANIFEST_DIR/rir.json"
if [[ ! -f "$RIR_MANIFEST" && -f "$MANIFEST_DIR/impulse_responses.json" ]]; then
  RIR_MANIFEST="$MANIFEST_DIR/impulse_responses.json"
  echo "Using impulse_responses.json as RIR manifest"
fi
check_free_space "$OUT_DIR" "$MIN_FREE_GB"
run_manifest_parallel "$RIR_MANIFEST" "rir" "rir" "$OUT_DIR/.manifest_shards/rir" "$RIR_SHARDS" "$RIR_WORKERS" "$RIR_LIMIT"

BG_MANIFEST="$MANIFEST_DIR/background.json"
if [[ -f "$BG_MANIFEST" ]]; then
  check_free_space "$OUT_DIR" "$MIN_FREE_GB"
  run_manifest_parallel "$BG_MANIFEST" "background_speech" "speech" "$OUT_DIR/.manifest_shards/background_speech" "$BG_SHARDS" "$BG_WORKERS" "$BG_LIMIT"
else
  echo "Skipping missing background manifest: $BG_MANIFEST"
fi

echo "Writing dataset.cfg"
"$PYTHON_BIN" - "$OUT_DIR" "$TRAIN_SPEECH_FACTOR" "$BACKGROUND_SPEECH_FACTOR" "$VALID_BACKGROUND_FACTOR" "$INCLUDE_BG_IN_VALID_TEST" "$NOISE_FACTOR" "$RIR_FACTOR" <<'PY'
import json
import os
import sys
from glob import glob

out_dir = sys.argv[1]
train_speech_factor = int(sys.argv[2])
background_speech_factor = int(sys.argv[3])
valid_background_factor = int(sys.argv[4])
include_bg_in_valid_test = int(sys.argv[5]) != 0
noise_factor = int(sys.argv[6])
rir_factor = int(sys.argv[7])

def has(name: str) -> bool:
    return os.path.isfile(os.path.join(out_dir, name))

train = []
valid = []
test = []

train_files = sorted([os.path.basename(p) for p in glob(os.path.join(out_dir, "train_speech*.hdf5"))])
val_files = sorted([os.path.basename(p) for p in glob(os.path.join(out_dir, "val_speech*.hdf5"))])
bg_files = sorted([os.path.basename(p) for p in glob(os.path.join(out_dir, "background_speech*.hdf5"))])
noise_files = sorted([os.path.basename(p) for p in glob(os.path.join(out_dir, "noise*.hdf5"))])
rir_files = sorted([os.path.basename(p) for p in glob(os.path.join(out_dir, "rir*.hdf5"))])

if has("train_speech.hdf5") and "train_speech.hdf5" not in train_files:
    train_files.insert(0, "train_speech.hdf5")
if has("val_speech.hdf5") and "val_speech.hdf5" not in val_files:
    val_files.insert(0, "val_speech.hdf5")
if has("noise.hdf5") and "noise.hdf5" not in noise_files:
    noise_files.insert(0, "noise.hdf5")
if has("rir.hdf5") and "rir.hdf5" not in rir_files:
    rir_files.insert(0, "rir.hdf5")

for tr in train_files:
    train.append([tr, train_speech_factor])
for bg in bg_files:
    train.append([bg, background_speech_factor])
for ns in noise_files:
    train.append([ns, noise_factor])
for rr in rir_files:
    train.append([rr, rir_factor])

for v in val_files:
    valid.append([v, train_speech_factor])
    test.append([v, train_speech_factor])
if include_bg_in_valid_test:
    for bg in bg_files:
        valid.append([bg, valid_background_factor])
        test.append([bg, valid_background_factor])
for ns in noise_files:
    valid.append([ns, noise_factor])
    test.append([ns, noise_factor])
for rr in rir_files:
    valid.append([rr, rir_factor])
    test.append([rr, rir_factor])

cfg = {"train": train, "valid": valid, "test": test}

with open(os.path.join(out_dir, "dataset.cfg"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

print("dataset.cfg written:")
print(json.dumps(cfg, indent=2))
PY

cleanup_generated_artifacts "$CLEANUP_MODE"
echo "HDF5 preparation complete."
