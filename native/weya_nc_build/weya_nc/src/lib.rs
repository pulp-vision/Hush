//! Weya Noise Cancellation Library
//! 
//! A C-compatible FFI wrapper around DeepFilterNet for real-time noise suppression.
//! Designed for high-concurrency phone call processing.
//!
//! # Architecture
//! - Model: Loaded once per process, shared across all calls (~5MB)
//! - Session: Created per call, isolated state (~50KB per call)
//!
//! # Resampling Strategy
//! Uses simple linear interpolation for resampling to maintain exact sample counts.
//! This avoids the sample count drift issues with rubato's sinc interpolation.

use std::ffi::{c_char, c_float, CStr};
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;

use df::tract::{DfModelShared, DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

/// Shared model handle.
///
/// Holds the compiled DeepFilterNet model (`DfModelShared`) once per process.
/// The model plans/weights are immutable and `Send + Sync`, so every session
/// shares this single copy via `Arc` — no per-session weight duplication — while
/// each session keeps its own (non-shared) execution state.
pub struct WeyaModel {
    shared: Arc<DfModelShared>,
}

fn make_runtime_params(atten_lim_db: c_float) -> RuntimeParams {
    RuntimeParams::new(
        1,         // n_ch
        false,     // post_filter
        atten_lim_db,
        -15.0f32,  // min_db_thresh
        35.0f32,   // max_db_erb_thresh
        35.0f32,   // max_db_df_thresh
        ReduceMask::MAX,
    )
}

/// Load a model bundle and compile it once into a shared, thread-safe model.
fn load_shared_model(path: Option<&str>) -> Result<DfModelShared, String> {
    let params = match path {
        Some(p) => DfParams::new(PathBuf::from(p))
            .map_err(|e| format!("Could not load model bundle at '{}': {}", p, e))?,
        // No model is embedded in this build — the stock 48 kHz DeepFilterNet3
        // bundle and the features that pulled it in are removed from the vendored
        // df crate outright. The fine-tuned 16 kHz bundle must always be supplied
        // explicitly: set WEYA_NC_MODEL_PATH or call weya_nc_model_load_from_path.
        None => {
            return Err(
                "No model path provided and this library has no embedded model. \
                 Set WEYA_NC_MODEL_PATH or use weya_nc_model_load_from_path with the \
                 fine-tuned advanced_dfnet16k bundle."
                    .to_string(),
            )
        }
    };
    // Build the compiled model once. n_ch=1 (mono) and MAX reduce mask match the
    // per-session runtime params used in build_session_df below.
    DfModelShared::new(params, 1, ReduceMask::MAX)
        .map_err(|e| format!("Could not initialize DeepFilterNet model: {}", e))
}

/// Build a per-session DfTract that SHARES the immutable model plans via Arc.
/// No weights are copied — only fresh per-session execution state is allocated —
/// and no `Rc` state is shared across sessions/threads.
fn build_session_df(shared: &DfModelShared, atten_lim_db: c_float) -> Result<DfTract, String> {
    let r_params = make_runtime_params(100.0);
    let mut df = DfTract::from_shared(shared, &r_params)
        .map_err(|e| format!("Could not initialize DeepFilterNet runtime: {}", e))?;
    df.atten_lim = if atten_lim_db < 100.0 {
        Some(10f32.powf(-atten_lim_db / 20.0))
    } else {
        None
    };
    Ok(df)
}

/// Per-call session with isolated state — no shared mutable state with other sessions.
pub struct WeyaSession {
    df: DfTract,
    shared: Arc<DfModelShared>,
    atten_lim_db: c_float,
    hop_size: usize,      // Model hop size
    model_sr: usize,      // Model sample rate
    input_sr: usize,      // Input sample rate (e.g., 8000)
    input_hop: usize,     // Input frame size

    // Pre-allocated buffers
    input_48k: Array2<f32>,
    output_48k: Array2<f32>,
    
    // Resampling buffer (used for both upsampling and downsampling)
    resample_buf: Vec<f32>,

    // Frame counter for observability/debugging
    frame_count: usize,
}

/// Simple linear interpolation resampling
fn resample_linear(input: &[f32], input_sr: usize, output_sr: usize, output: &mut [f32]) {
    if input_sr == output_sr {
        output.copy_from_slice(input);
        return;
    }
    
    let n_in = input.len();
    let n_out = output.len();
    
    if n_in == 0 || n_out == 0 {
        output.fill(0.0);
        return;
    }
    
    let ratio = (n_in - 1) as f64 / (n_out - 1).max(1) as f64;
    
    for i in 0..n_out {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        
        if src_idx + 1 < n_in {
            // Linear interpolation between two samples
            output[i] = (input[src_idx] as f64 * (1.0 - frac) + 
                        input[src_idx + 1] as f64 * frac) as f32;
        } else if src_idx < n_in {
            output[i] = input[src_idx];
        } else {
            output[i] = 0.0;
        }
    }
}

/// Load the fine-tuned model bundle named by `WEYA_NC_MODEL_PATH`.
///
/// No model is embedded in this library, so this fails (returning null) unless
/// that environment variable points at a bundle. Prefer
/// `weya_nc_model_load_from_path` where the caller knows the path.
#[no_mangle]
pub extern "C" fn weya_nc_model_load() -> *mut WeyaModel {
    let env_model_path = std::env::var("WEYA_NC_MODEL_PATH").ok();
    let path = env_model_path.as_deref();
    match load_shared_model(path) {
        Ok(shared) => {
            let model = Box::new(WeyaModel {
                shared: Arc::new(shared),
            });
            Box::into_raw(model)
        }
        Err(e) => {
            eprintln!("[weya_nc] Error loading model: {}", e);
            ptr::null_mut()
        }
    }
}

/// Load model from an explicit ONNX tar.gz model bundle path.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_model_load_from_path(path: *const c_char) -> *mut WeyaModel {
    if path.is_null() {
        eprintln!("[weya_nc] Error: model path is null");
        return ptr::null_mut();
    }
    let c_str = CStr::from_ptr(path);
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[weya_nc] Error parsing model path utf-8: {}", e);
            return ptr::null_mut();
        }
    };
    match load_shared_model(Some(path_str)) {
        Ok(shared) => {
            let model = Box::new(WeyaModel {
                shared: Arc::new(shared),
            });
            Box::into_raw(model)
        }
        Err(e) => {
            eprintln!("[weya_nc] Error loading model from path: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free a model handle.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_model_free(model: *mut WeyaModel) {
    if !model.is_null() {
        drop(Box::from_raw(model));
    }
}

/// Create a new processing session.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_session_create(
    model: *const WeyaModel,
    input_sr: usize,
    atten_lim_db: c_float,
) -> *mut WeyaSession {
    if model.is_null() {
        eprintln!("[weya_nc] Error: model is null");
        return ptr::null_mut();
    }
    
    let model_ref = &*model;

    // Build a session that shares the immutable model plans via Arc (no weight
    // copy) and owns its fresh, independent execution state.
    let df = match build_session_df(&model_ref.shared, atten_lim_db) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[weya_nc] Error creating session: {}", e);
            return ptr::null_mut();
        }
    };

    let model_sr = df.sr;
    let hop_size = df.hop_size;

    // Calculate input hop size for resampling
    let input_hop = if input_sr != model_sr {
        ((hop_size as f64) * (input_sr as f64) / (model_sr as f64)).round() as usize
    } else {
        hop_size
    };

    // Pre-allocate buffers
    let input_48k = Array2::<f32>::zeros((1, hop_size));
    let output_48k = Array2::<f32>::zeros((1, hop_size));
    let resample_buf = vec![0.0f32; hop_size];

    let session = Box::new(WeyaSession {
        df,
        shared: Arc::clone(&model_ref.shared),
        atten_lim_db,
        hop_size,
        model_sr,
        input_sr,
        input_hop,
        input_48k,
        output_48k,
        resample_buf,
        frame_count: 0,
    });
    
    Box::into_raw(session)
}

/// Free a session handle.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_session_free(session: *mut WeyaSession) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}

/// Get the expected input frame length in samples.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_get_frame_length(session: *const WeyaSession) -> usize {
    if session.is_null() { return 0; }
    (*session).input_hop
}

/// Get the model sample rate (48kHz).
#[no_mangle]
pub unsafe extern "C" fn weya_nc_get_sample_rate(session: *const WeyaSession) -> usize {
    if session.is_null() { return 0; }
    (*session).model_sr
}

/// Get the input sample rate configured for this session.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_get_input_sample_rate(session: *const WeyaSession) -> usize {
    if session.is_null() { return 0; }
    (*session).input_sr
}

/// Process one audio frame.
///
/// Uses linear interpolation for resampling to maintain exact sample counts
/// and avoid frame-to-frame drift.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_process_frame(
    session: *mut WeyaSession,
    input: *const c_float,
    output: *mut c_float,
) -> c_float {
    if session.is_null() || input.is_null() || output.is_null() {
        return -100.0;
    }
    
    let sess = &mut *session;
    let frame_len = sess.input_hop;
    
    let input_slice = std::slice::from_raw_parts(input, frame_len);
    let output_slice = std::slice::from_raw_parts_mut(output, frame_len);
    
    sess.frame_count += 1;
    
    // Check if we need resampling
    if sess.input_sr != sess.model_sr {
        // ========================================
        // STEP 1: Upsample input_hop -> hop_size using linear interpolation
        // ========================================
        resample_linear(input_slice, sess.input_sr, sess.model_sr, &mut sess.resample_buf);
        
        // Copy to 2D array for DfTract
        for i in 0..sess.hop_size {
            sess.input_48k[[0, i]] = sess.resample_buf[i];
        }
        
        // ========================================  
        // STEP 2: Process through DeepFilterNet
        // ========================================
        let noisy_view: ArrayView2<f32> = sess.input_48k.view();
        let enh_view: ArrayViewMut2<f32> = sess.output_48k.view_mut();
        
        let lsnr = match sess.df.process(noisy_view, enh_view) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[weya_nc] Process error: {}", e);
                // On error, passthrough original
                output_slice.copy_from_slice(input_slice);
                return -100.0;
            }
        };
        
        // ========================================
        // STEP 3: Downsample hop_size -> input_hop using linear interpolation
        // ========================================
        // Copy from 2D array
        for i in 0..sess.hop_size {
            sess.resample_buf[i] = sess.output_48k[[0, i]];
        }
        
        resample_linear(&sess.resample_buf, sess.model_sr, sess.input_sr, output_slice);
        
        lsnr
    } else {
        // No resampling needed - direct processing at 48kHz
        for (i, &v) in input_slice.iter().enumerate() {
            sess.input_48k[[0, i]] = v;
        }
        
        let noisy_view: ArrayView2<f32> = sess.input_48k.view();
        let enh_view: ArrayViewMut2<f32> = sess.output_48k.view_mut();
        
        let lsnr = match sess.df.process(noisy_view, enh_view) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[weya_nc] Process error: {}", e);
                output_slice.copy_from_slice(input_slice);
                return -100.0;
            }
        };
        
        for (i, v) in output_slice.iter_mut().enumerate() {
            *v = sess.output_48k[[0, i]];
        }
        
        lsnr
    }
}

/// Reset session state for a new audio stream.
#[no_mangle]
pub unsafe extern "C" fn weya_nc_reset(session: *mut WeyaSession) {
    if !session.is_null() {
        let sess = &mut *session;
        match build_session_df(&sess.shared, sess.atten_lim_db) {
            Ok(df) => sess.df = df,
            Err(e) => {
                eprintln!("[weya_nc] Error resetting session: {}", e);
                return;
            }
        }
        sess.input_48k.fill(0.0);
        sess.output_48k.fill(0.0);
        sess.resample_buf.fill(0.0);
        sess.frame_count = 0;
    }
}
