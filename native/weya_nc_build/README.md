# weya_nc build source

Buildable source of truth for the `weya_nc` noise-cancellation shared library
(the `libweya_nc.{so,dylib}` / `weya_nc.dll` shipped in **Hush** `deployment/lib/`
and consumed by **vocalyst** `components/noise_cancellation/lib/`).

Both Hush and vocalyst ship only the **prebuilt** binaries — this directory is
where they are built.

## Layout

- `weya_nc/` — the C-ABI cdylib crate (FFI wrapper around DeepFilterNet).
- `vendor/deep_filter/` — a trimmed vendored copy of **DeepFilterNet `df` v0.5.3**
  with the shared-weights patch in `src/tract.rs`.

## Two key properties

1. **Single shared model.** `src/tract.rs` exposes `DfModelShared` +
   `DfTract::from_shared`, so every session shares the compiled model
   (`Arc<TypedSimplePlan>`, `Send + Sync`) instead of copying the weights. Fixes
   the issue-#5 concurrent-crash (no shared `Rc<Tensor>`) AND keeps per-session
   memory to a few KB.
2. **No embedded model — removed, not just disabled.** The stock 48 kHz
   DeepFilterNet3 bundle is gone from the vendored crate entirely: the
   `default-model` / `default-model-ll` features, the `capi` feature that pulled
   them in, and the `impl Default for DfParams` / `impl Default for DfTract`
   blocks that `include_bytes!`d the model have all been deleted. There is no
   feature flag that can put a model back in the binary. The library runs **only**
   the fine-tuned bundle supplied at runtime:
   `advanced_dfnet16k_model_best_onnx.tar.gz` (via `weya_nc_model_load_from_path`
   or `WEYA_NC_MODEL_PATH`), and returns null with an explicit error if none is
   given — it never silently falls back.

## Build

```bash
cd weya_nc

# Linux x86_64 (native)
cargo build --release
#  -> target/release/libweya_nc.so

# Windows x86_64 (mingw: x86_64-w64-mingw32-gcc)
cargo build --release --target x86_64-pc-windows-gnu
#  -> target/x86_64-pc-windows-gnu/release/weya_nc.dll

# macOS arm64 (osxcross, via Docker)
docker run --rm -v "$PWD/..":/work joseluisq/rust-linux-darwin-builder:latest bash -c '
  export PATH=/usr/local/osxcross/target/bin:$PATH
  export CC_aarch64_apple_darwin=/usr/local/osxcross/target/bin/oa64-clang
  export CXX_aarch64_apple_darwin=/usr/local/osxcross/target/bin/oa64-clang++
  export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER=/usr/local/osxcross/target/bin/oa64-clang
  export CARGO_TARGET_DIR=/work/target-mac
  cd /work/weya_nc && cargo build --release --target aarch64-apple-darwin'
#  -> ../target-mac/aarch64-apple-darwin/release/libweya_nc.dylib
```

Each binary is ~10–15 MB (no embedded model) and exports the 10 `weya_nc_*` C
symbols.

## Validate

```bash
WEYA_NC_LIB_PATH=target/release/libweya_nc.so \
WEYA_NC_MODEL_PATH=/path/to/advanced_dfnet16k_model_best_onnx.tar.gz \
python3 /home/ubuntu/Hush/deployment/examples/python/test_concurrency.py
```

## Distribute

Copy the built binaries to:

- `Hush/deployment/lib/{libweya_nc.so,libweya_nc.dylib,weya_nc.dll}`
- `vocalyst/components/noise_cancellation/lib/{libweya_nc.so,libweya_nc.dylib,weya_nc.dll}`
