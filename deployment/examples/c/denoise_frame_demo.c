#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../include/weya_nc.h"

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s <model.tar.gz> <input_i16_mono.pcm> <output_i16_mono.pcm> [input_sr]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* in_path = argv[2];
    const char* out_path = argv[3];
    size_t input_sr = (argc == 5) ? (size_t)strtoull(argv[4], NULL, 10) : 16000;

    WeyaModel* model = weya_nc_model_load_from_path(model_path);
    if (model == NULL) {
        fprintf(stderr, "Failed to load model from: %s\n", model_path);
        return 1;
    }

    WeyaSession* session = weya_nc_session_create(model, input_sr, 100.0f);
    if (session == NULL) {
        fprintf(stderr, "Failed to create session\n");
        weya_nc_model_free(model);
        return 1;
    }

    size_t frame_len = weya_nc_get_frame_length(session);
    int16_t* in_i16 = (int16_t*)calloc(frame_len, sizeof(int16_t));
    int16_t* out_i16 = (int16_t*)calloc(frame_len, sizeof(int16_t));
    float* in_f32 = (float*)calloc(frame_len, sizeof(float));
    float* out_f32 = (float*)calloc(frame_len, sizeof(float));
    if (!in_i16 || !out_i16 || !in_f32 || !out_f32) {
        fprintf(stderr, "Allocation failure\n");
        weya_nc_session_free(session);
        weya_nc_model_free(model);
        free(in_i16);
        free(out_i16);
        free(in_f32);
        free(out_f32);
        return 1;
    }

    FILE* fin = fopen(in_path, "rb");
    FILE* fout = fopen(out_path, "wb");
    if (!fin || !fout) {
        fprintf(stderr, "Could not open input/output files\n");
        if (fin) fclose(fin);
        if (fout) fclose(fout);
        weya_nc_session_free(session);
        weya_nc_model_free(model);
        free(in_i16);
        free(out_i16);
        free(in_f32);
        free(out_f32);
        return 1;
    }

    while (1) {
        size_t n = fread(in_i16, sizeof(int16_t), frame_len, fin);
        if (n == 0) break;

        for (size_t i = 0; i < frame_len; i++) {
            in_f32[i] = (i < n) ? ((float)in_i16[i] / 32768.0f) : 0.0f;
        }

        (void)weya_nc_process_frame(session, in_f32, out_f32);

        for (size_t i = 0; i < n; i++) {
            float v = clampf(out_f32[i] * 32768.0f, -32768.0f, 32767.0f);
            out_i16[i] = (int16_t)v;
        }
        fwrite(out_i16, sizeof(int16_t), n, fout);
    }

    fclose(fin);
    fclose(fout);
    weya_nc_session_free(session);
    weya_nc_model_free(model);
    free(in_i16);
    free(out_i16);
    free(in_f32);
    free(out_f32);

    printf("Denoising complete: %s\n", out_path);
    return 0;
}
