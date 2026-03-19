#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUST_DIR="$ROOT_DIR/rust"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"

echo "[1/3] Building Rust workspace (hdf5_prep)..."
cargo build --release -p hdf5_prep --manifest-path "$RUST_DIR/Cargo.toml"

echo "[2/3] Installing libdf Python extension..."
if [[ "$FORCE_REINSTALL" != "1" ]] && "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import libdf
PY
then
  echo "libdf already importable in $PYTHON_BIN; skipping install."
else
  if [[ "$FORCE_REINSTALL" == "1" ]]; then
    echo "FORCE_REINSTALL=1 -> rebuilding libdf"
  fi
  "$PYTHON_BIN" -m pip install --no-build-isolation -e "$RUST_DIR/pyDF"
fi

echo "[3/3] Installing libdfdata Python extension..."
if [[ "$FORCE_REINSTALL" != "1" ]] && "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import libdfdata
PY
then
  echo "libdfdata already importable in $PYTHON_BIN; skipping install."
else
  if [[ "$FORCE_REINSTALL" == "1" ]]; then
    echo "FORCE_REINSTALL=1 -> rebuilding libdfdata"
  fi
  "$PYTHON_BIN" -m pip install --no-build-isolation -e "$RUST_DIR/pyDF-data"
fi

echo "Rust data stack is ready."
