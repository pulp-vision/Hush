#ifndef WEYA_NC_H
#define WEYA_NC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WeyaModel WeyaModel;
typedef struct WeyaSession WeyaSession;

// Load default model, or model from WEYA_NC_MODEL_PATH if env var is set.
WeyaModel* weya_nc_model_load(void);

// Load ONNX tar.gz bundle from explicit path.
WeyaModel* weya_nc_model_load_from_path(const char* path);

// Free model handle.
void weya_nc_model_free(WeyaModel* model);

// Create a new session.
WeyaSession* weya_nc_session_create(const WeyaModel* model, size_t input_sr, float atten_lim_db);

// Free session.
void weya_nc_session_free(WeyaSession* session);

// Query frame parameters.
size_t weya_nc_get_frame_length(const WeyaSession* session);
size_t weya_nc_get_sample_rate(const WeyaSession* session);
size_t weya_nc_get_input_sample_rate(const WeyaSession* session);

// Process one frame of normalized float32 audio in range [-1.0, 1.0].
float weya_nc_process_frame(WeyaSession* session, const float* input, float* output);

// Reset streaming state for a new audio stream.
void weya_nc_reset(WeyaSession* session);

#ifdef __cplusplus
}
#endif

#endif  // WEYA_NC_H
