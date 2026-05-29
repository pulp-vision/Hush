#!/usr/bin/env python3
"""Concurrency stress test / regression test for the Weya NC wrapper.

This reproduces the usage pattern reported in issue #5 — many threads creating
sessions, processing frames, resetting and freeing concurrently — and verifies
that the (now thread-safe) library:

  1. never crashes the process (before the fix it corrupted the heap under this
     load: ``malloc(): unsorted double linked list corrupted``), and
  2. produces output identical to a single-threaded reference run, proving
     concurrent sessions don't interfere with each other.

Run directly::

    python3 deployment/examples/python/test_concurrency.py

or under pytest::

    pytest deployment/examples/python/test_concurrency.py

It is skipped automatically when the library or model bundle is not available.
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weya_nc import WeyaModel, _find_library, _find_model  # noqa: E402

N_THREADS = 16
N_SESSIONS_PER_THREAD = 8
N_FRAMES = 50
FRAME_LEN = 160

# One shared model handle, used by every thread — this is exactly the pattern
# reported in issue #5 that corrupted the heap before the fix.
_MODEL: "WeyaModel | None" = None


def _available() -> bool:
    try:
        _find_library(None)
        _find_model(None)
        return True
    except FileNotFoundError:
        return False


def _make_input(seed: int, frame_len: int) -> np.ndarray:
    """Deterministic noisy input: one row per frame."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N_FRAMES, frame_len)) * 0.1).astype(np.float32)


def _run_one_session(frames: np.ndarray) -> np.ndarray:
    """Full session lifecycle: create -> process -> reset -> process -> close.

    Returns the denoised output of the post-reset pass, stacked per frame.
    """
    nc = _MODEL.create_session()
    try:
        # First pass (state warms up), then reset and re-run the same frames so
        # the output depends only on the input — not on prior session history.
        for f in frames:
            nc.process_frame(f)
        nc.reset()
        out = np.stack([nc.process_frame(f).copy() for f in frames])
    finally:
        nc.close()
    return out


def _reference(frames: np.ndarray) -> np.ndarray:
    return _run_one_session(frames)


def test_concurrent_sessions_match_reference() -> None:
    global _MODEL
    if not _available():
        print("SKIP: libweya_nc / model bundle not found")
        return

    _MODEL = WeyaModel()

    # Each "slot" has its own deterministic input so we can verify correctness.
    inputs = [_make_input(seed, FRAME_LEN) for seed in range(N_SESSIONS_PER_THREAD)]
    references = [_reference(inp) for inp in inputs]

    errors: list[str] = []

    def worker(thread_id: int) -> None:
        try:
            for i, inp in enumerate(inputs):
                out = _run_one_session(inp)
                if not np.array_equal(out, references[i]):
                    max_diff = float(np.max(np.abs(out - references[i])))
                    errors.append(
                        f"thread {thread_id} slot {i}: output mismatch "
                        f"(max abs diff {max_diff})"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"thread {thread_id}: {exc!r}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futures = [ex.submit(worker, t) for t in range(N_THREADS)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
    finally:
        _MODEL.close()

    assert not errors, "Concurrency failures:\n" + "\n".join(errors)

    total = N_THREADS * N_SESSIONS_PER_THREAD
    print(
        f"OK: {total} sessions across {N_THREADS} threads "
        f"({total * N_FRAMES * 2} frames) — no crash, outputs match reference."
    )


# High-contention churn: many threads rapidly create / process / reset / free
# sessions on a shared model with minimal Python work in between, maximizing the
# overlap of native calls. This is the configuration that reliably triggered
# ``malloc(): unsorted double linked list corrupted`` on the pre-fix library, so
# it is the actual regression guard for the fix.
CHURN_THREADS = 48
CHURN_SESSIONS_PER_THREAD = 60


def test_session_churn_does_not_crash() -> None:
    global _MODEL
    if not _available():
        print("SKIP: libweya_nc / model bundle not found")
        return

    _MODEL = WeyaModel()
    frame = (np.random.default_rng(0).standard_normal(FRAME_LEN) * 0.1).astype(
        np.float32
    )
    errors: list[str] = []

    def worker(thread_id: int) -> None:
        try:
            for _ in range(CHURN_SESSIONS_PER_THREAD):
                nc = _MODEL.create_session()
                for _ in range(4):
                    nc.process_frame(frame)
                nc.reset()
                nc.process_frame(frame)
                nc.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"thread {thread_id}: {exc!r}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CHURN_THREADS) as ex:
            futures = [ex.submit(worker, t) for t in range(CHURN_THREADS)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
    finally:
        _MODEL.close()

    assert not errors, "Churn failures:\n" + "\n".join(errors)

    total = CHURN_THREADS * CHURN_SESSIONS_PER_THREAD
    print(
        f"OK: {total} session create/free cycles across {CHURN_THREADS} threads "
        "— no crash."
    )


if __name__ == "__main__":
    test_concurrent_sessions_match_reference()
    test_session_churn_does_not_crash()
