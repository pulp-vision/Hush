"""Reusable Python wrapper for the Weya NC Standalone C library.

Provides a clean interface to the 10-function C API exposed by libweya_nc.
Both ``denoise_mic.py`` and ``denoise_stream.py`` import this module.

Usage::

    from weya_nc import WeyaNC

    with WeyaNC(lib_path="deployment/lib/libweya_nc.so",
                model_path="deployment/models/advanced_dfnet16k_model_best_onnx.tar.gz") as nc:
        clean = nc.process_frame(noisy_frame)

Thread safety
-------------
The native library is thread-safe across sessions: any number of sessions may
be created, run, reset and freed from different threads **at the same time**,
and they run in parallel. The one rule is that a *single* session is not
reentrant — use one session per concurrent stream/thread. This wrapper takes a
small per-instance lock around :meth:`WeyaNC.process_frame`/:meth:`WeyaNC.reset`
so that even accidental concurrent calls on the same object stay safe.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# Guards only the shared library cache below — *not* the native calls, which are
# thread-safe across sessions in the current library (issue #5 fix).
_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

_LIB_NAMES = {
    "Linux": "libweya_nc.so",
    "Darwin": "libweya_nc.dylib",
    "Windows": "weya_nc.dll",
}


def _platform_lib_name() -> str:
    return _LIB_NAMES.get(platform.system(), "libweya_nc.so")


def _repo_root() -> Path:
    """Best-effort repo root: three levels up from this file (deployment/examples/python/)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _find_library(explicit: Optional[str | Path]) -> Path:
    """Resolve library path with fallback search order."""
    if explicit is not None:
        p = Path(explicit).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Library not found at explicit path: {p}\n"
            "Download it from the GitHub Releases page and place it at "
            "deployment/lib/"
        )

    name = _platform_lib_name()

    # Relative to repo
    repo_lib = _repo_root() / "deployment" / "lib" / name
    if repo_lib.exists():
        return repo_lib

    # Environment variable
    env = os.environ.get("WEYA_NC_LIB_PATH")
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find {name}.\n"
        "Either:\n"
        f"  1. Place it at {repo_lib}\n"
        "  2. Set the WEYA_NC_LIB_PATH environment variable\n"
        "  3. Pass lib_path= explicitly\n"
        "Download from the GitHub Releases page."
    )


def _find_model(explicit: Optional[str | Path]) -> Path:
    """Resolve model path with fallback search order."""
    if explicit is not None:
        p = Path(explicit).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Model not found at explicit path: {p}\n"
            "Download it from the GitHub Releases page and place it at "
            "deployment/models/"
        )

    # Relative to repo
    repo_model = _repo_root() / "deployment" / "models" / "advanced_dfnet16k_model_best_onnx.tar.gz"
    if repo_model.exists():
        return repo_model

    # Environment variable
    env = os.environ.get("WEYA_NC_MODEL_PATH")
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find ONNX model bundle.\n"
        "Either:\n"
        f"  1. Place it at {repo_model}\n"
        "  2. Set the WEYA_NC_MODEL_PATH environment variable\n"
        "  3. Pass model_path= explicitly\n"
        "Download from the GitHub Releases page."
    )


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

# One ``dlopen`` per library path is enough — repeated loads return the same
# in-process handle anyway. Caching also means we declare the C signatures once.
_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def _declare_api(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for every exported function."""
    lib.weya_nc_model_load.argtypes = []
    lib.weya_nc_model_load.restype = ctypes.c_void_p

    lib.weya_nc_model_load_from_path.argtypes = [ctypes.c_char_p]
    lib.weya_nc_model_load_from_path.restype = ctypes.c_void_p

    lib.weya_nc_model_free.argtypes = [ctypes.c_void_p]
    lib.weya_nc_model_free.restype = None

    lib.weya_nc_session_create.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float,
    ]
    lib.weya_nc_session_create.restype = ctypes.c_void_p

    lib.weya_nc_session_free.argtypes = [ctypes.c_void_p]
    lib.weya_nc_session_free.restype = None

    lib.weya_nc_get_frame_length.argtypes = [ctypes.c_void_p]
    lib.weya_nc_get_frame_length.restype = ctypes.c_size_t

    lib.weya_nc_get_sample_rate.argtypes = [ctypes.c_void_p]
    lib.weya_nc_get_sample_rate.restype = ctypes.c_size_t

    lib.weya_nc_get_input_sample_rate.argtypes = [ctypes.c_void_p]
    lib.weya_nc_get_input_sample_rate.restype = ctypes.c_size_t

    float_ptr = ctypes.POINTER(ctypes.c_float)
    lib.weya_nc_process_frame.argtypes = [ctypes.c_void_p, float_ptr, float_ptr]
    lib.weya_nc_process_frame.restype = ctypes.c_float

    lib.weya_nc_reset.argtypes = [ctypes.c_void_p]
    lib.weya_nc_reset.restype = None


def _load_library(lib_file: Path) -> ctypes.CDLL:
    key = str(lib_file)
    with _CACHE_LOCK:
        lib = _LIB_CACHE.get(key)
        if lib is None:
            lib = ctypes.CDLL(key)
            _declare_api(lib)
            _LIB_CACHE[key] = lib
        return lib


# ---------------------------------------------------------------------------
# Model handle
# ---------------------------------------------------------------------------

class WeyaModel:
    """A loaded model that can spawn many independent denoising sessions.

    Loading the ~8 MB model is expensive, so a server handling many concurrent
    streams should load it **once** and create one cheap session per stream::

        model = WeyaModel()                 # load once
        nc = model.create_session()         # one per call/stream

    Sessions created from one model are independent and may be created, used and
    freed from different threads concurrently — they run in parallel.

    Parameters
    ----------
    lib_path : str or Path, optional
        Explicit path to the shared library. Auto-discovered if *None*.
    model_path : str or Path, optional
        Explicit path to the ONNX model tar.gz bundle. Auto-discovered if *None*.
    """

    def __init__(
        self,
        lib_path: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
    ) -> None:
        lib_file = _find_library(lib_path)
        model_file = _find_model(model_path)

        self._lib = _load_library(lib_file)
        self._model = self._lib.weya_nc_model_load_from_path(
            str(model_file).encode("utf-8")
        )
        if not self._model:
            raise RuntimeError(f"Failed to load model from {model_file}")

    def create_session(
        self,
        sample_rate: int = 16000,
        atten_lim_db: float = 100.0,
    ) -> "WeyaNC":
        """Create a new streaming session backed by this shared model."""
        if not getattr(self, "_model", None):
            raise RuntimeError("Model has been closed")
        return WeyaNC(
            _model=self,
            sample_rate=sample_rate,
            atten_lim_db=atten_lim_db,
        )

    def close(self) -> None:
        """Free the native model handle. Sessions must be closed first."""
        if getattr(self, "_model", None):
            self._lib.weya_nc_model_free(self._model)
            self._model = None

    def __enter__(self) -> "WeyaModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# C API wrapper
# ---------------------------------------------------------------------------

class WeyaNC:
    """High-level wrapper around a single Weya NC denoising session.

    Parameters
    ----------
    lib_path : str or Path, optional
        Explicit path to the shared library. Auto-discovered if *None*.
        Ignored when *_model* is supplied.
    model_path : str or Path, optional
        Explicit path to the ONNX model tar.gz bundle. Auto-discovered if *None*.
        Ignored when *_model* is supplied.
    sample_rate : int
        Input audio sample rate (default 16000). The library resamples internally
        if this differs from the model's native 16 kHz.
    atten_lim_db : float
        Maximum attenuation in dB (default 100.0 = unlimited).
    _model : WeyaModel, optional
        Share an already-loaded :class:`WeyaModel` instead of loading a private
        one. Prefer ``WeyaModel(...).create_session(...)`` over passing this
        directly. When set, this session does **not** own/free the model.

    Notes
    -----
    Distinct sessions are independent and run in parallel across threads. A
    single session is not reentrant, so use one per concurrent stream; a small
    per-instance lock guards this object's scratch buffers against accidental
    concurrent use.
    """

    def __init__(
        self,
        lib_path: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        sample_rate: int = 16000,
        atten_lim_db: float = 100.0,
        _model: Optional[WeyaModel] = None,
    ) -> None:
        if _model is not None:
            self._model_owner = None  # shared; not owned by this session
            self._model = _model
        else:
            self._model_owner = WeyaModel(lib_path=lib_path, model_path=model_path)
            self._model = self._model_owner

        self._lib = self._model._lib
        model_handle = self._model._model

        # Guards this session's non-reentrant native handle and scratch buffers.
        # Distinct WeyaNC instances use distinct locks, so they run in parallel.
        self._lock = threading.Lock()

        self._session = self._lib.weya_nc_session_create(
            model_handle, sample_rate, ctypes.c_float(atten_lim_db)
        )
        if not self._session:
            raise RuntimeError("Failed to create denoising session")

        self._sample_rate = int(
            self._lib.weya_nc_get_input_sample_rate(self._session)
        )
        self._frame_length = int(
            self._lib.weya_nc_get_frame_length(self._session)
        )

        # Pre-allocated buffers
        self._buf_in = np.zeros(self._frame_length, dtype=np.float32)
        self._buf_out = np.zeros(self._frame_length, dtype=np.float32)

    # -- Properties ----------------------------------------------------------

    @property
    def frame_length(self) -> int:
        """Number of samples expected per ``process_frame`` call."""
        return self._frame_length

    @property
    def sample_rate(self) -> int:
        """Configured input sample rate."""
        return self._sample_rate

    # -- Processing ----------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Denoise a single audio frame.

        Accepts ``int16`` or ``float32`` input and returns the same dtype.
        The frame length must equal :pyattr:`frame_length`.
        """
        if len(frame) != self._frame_length:
            raise ValueError(
                f"Expected {self._frame_length} samples, got {len(frame)}"
            )

        input_dtype = frame.dtype
        float_ptr = ctypes.POINTER(ctypes.c_float)

        # Per-instance lock: protects this session's non-reentrant native handle
        # and its shared ``_buf_in``/``_buf_out`` scratch buffers. Other sessions
        # hold their own locks, so they process frames in parallel.
        with self._lock:
            if input_dtype == np.int16:
                self._buf_in[:] = frame.astype(np.float32) / 32768.0
            else:
                self._buf_in[:] = frame.astype(np.float32)

            self._lib.weya_nc_process_frame(
                self._session,
                self._buf_in.ctypes.data_as(float_ptr),
                self._buf_out.ctypes.data_as(float_ptr),
            )

            if input_dtype == np.int16:
                return (self._buf_out * 32768.0).clip(-32768, 32767).astype(np.int16)
            return self._buf_out.copy()

    def reset(self) -> None:
        """Reset streaming state for a new audio stream."""
        with self._lock:
            if self._session:
                self._lib.weya_nc_reset(self._session)

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release the session (and the model, if this session loaded it)."""
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if getattr(self, "_session", None):
                    self._lib.weya_nc_session_free(self._session)
                    self._session = None
        # Only free the model if this session created it; a shared WeyaModel is
        # owned by the caller.
        if getattr(self, "_model_owner", None) is not None:
            self._model_owner.close()
            self._model_owner = None
        self._model = None

    def __enter__(self) -> "WeyaNC":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
