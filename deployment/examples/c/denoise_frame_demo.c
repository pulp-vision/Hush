/*
 * Minimal C example: denoise a WAV file with libweya_nc.
 *
 * Build (macOS / Linux):
 *   cc -O2 -Wall -Wextra denoise_frame_demo.c \
 *      -I../../include -L../../lib -lweya_nc \
 *      -Wl,-rpath,@loader_path/../../lib   # macOS
 *      # -Wl,-rpath,'$ORIGIN/../../lib'    # Linux
 *      -o denoise_frame_demo
 *
 * Usage:
 *   ./denoise_frame_demo <model.tar.gz> <input.wav> <output.wav> [atten_lim_db]
 *
 * Accepts 16-bit PCM WAV, mono or stereo (auto-downmixed). Sample rate is
 * read from the WAV header and passed to the session, matching the Python
 * ctypes example.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/weya_nc.h"

/* ---------- minimal little-endian RIFF/WAVE reader ---------------------- */

typedef struct {
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bits_per_sample;
    uint32_t data_bytes;  /* size of the data chunk payload */
    int16_t* samples;     /* interleaved int16 samples, channels * frames */
    size_t   nframes;     /* per-channel frame count */
} WavData;

static int read_u32_le(FILE* f, uint32_t* out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_u16_le(FILE* f, uint16_t* out) {
    unsigned char b[2];
    if (fread(b, 1, 2, f) != 2) return 0;
    *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return 1;
}

static int wav_load(const char* path, WavData* w) {
    memset(w, 0, sizeof(*w));
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "wav_load: cannot open %s\n", path);
        return 0;
    }

    char riff[4], wave[4];
    uint32_t riff_size;
    if (fread(riff, 1, 4, f) != 4 || !read_u32_le(f, &riff_size) ||
        fread(wave, 1, 4, f) != 4) {
        fprintf(stderr, "wav_load: header read failed\n");
        fclose(f); return 0;
    }
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "wav_load: not a RIFF/WAVE file\n");
        fclose(f); return 0;
    }

    uint16_t fmt_tag = 0, channels = 0, bits = 0, block_align = 0;
    uint32_t sr = 0, byte_rate = 0;
    int got_fmt = 0;

    /* Walk chunks until we find "data". */
    while (1) {
        char id[4];
        uint32_t sz;
        if (fread(id, 1, 4, f) != 4 || !read_u32_le(f, &sz)) {
            fprintf(stderr, "wav_load: unexpected EOF in chunk header\n");
            fclose(f); return 0;
        }

        if (memcmp(id, "fmt ", 4) == 0) {
            long fmt_start = ftell(f);
            if (!read_u16_le(f, &fmt_tag) || !read_u16_le(f, &channels) ||
                !read_u32_le(f, &sr)      || !read_u32_le(f, &byte_rate) ||
                !read_u16_le(f, &block_align) || !read_u16_le(f, &bits)) {
                fprintf(stderr, "wav_load: fmt chunk truncated\n");
                fclose(f); return 0;
            }
            /* Skip any extension bytes. */
            long consumed = ftell(f) - fmt_start;
            if ((long)sz > consumed) fseek(f, sz - consumed, SEEK_CUR);
            if (sz & 1) fseek(f, 1, SEEK_CUR);  /* RIFF pad */
            got_fmt = 1;
        } else if (memcmp(id, "data", 4) == 0) {
            if (!got_fmt) {
                fprintf(stderr, "wav_load: data chunk before fmt\n");
                fclose(f); return 0;
            }
            if (fmt_tag != 1) {
                fprintf(stderr, "wav_load: only PCM (fmt_tag=1) supported, got %u\n", fmt_tag);
                fclose(f); return 0;
            }
            if (bits != 16) {
                fprintf(stderr, "wav_load: only 16-bit PCM supported, got %u-bit\n", bits);
                fclose(f); return 0;
            }
            if (channels < 1 || channels > 2) {
                fprintf(stderr, "wav_load: only mono/stereo supported, got %u channels\n", channels);
                fclose(f); return 0;
            }

            size_t total = sz / sizeof(int16_t);
            int16_t* buf = (int16_t*)malloc(total * sizeof(int16_t));
            if (!buf) { fprintf(stderr, "wav_load: OOM\n"); fclose(f); return 0; }
            if (fread(buf, sizeof(int16_t), total, f) != total) {
                fprintf(stderr, "wav_load: data read truncated\n");
                free(buf); fclose(f); return 0;
            }

            w->sample_rate = sr;
            w->channels = channels;
            w->bits_per_sample = bits;
            w->data_bytes = sz;
            w->samples = buf;
            w->nframes = total / channels;
            fclose(f);
            return 1;
        } else {
            /* Skip unknown chunk + RIFF pad byte. */
            fseek(f, sz + (sz & 1), SEEK_CUR);
        }
    }
}

static int wav_write_mono_i16(const char* path, const int16_t* samples,
                              size_t nframes, uint32_t sr) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "wav_write: cannot open %s\n", path); return 0; }

    uint32_t data_bytes = (uint32_t)(nframes * sizeof(int16_t));
    uint32_t fmt_size = 16;
    uint16_t channels = 1, bits = 16, block_align = 2, fmt_tag = 1;
    uint32_t byte_rate = sr * channels * (bits / 8);
    uint32_t riff_size = 4 + (8 + fmt_size) + (8 + data_bytes);

    unsigned char hdr[44];
    memcpy(hdr +  0, "RIFF", 4);
    hdr[ 4] =  riff_size        & 0xff;
    hdr[ 5] = (riff_size >>  8) & 0xff;
    hdr[ 6] = (riff_size >> 16) & 0xff;
    hdr[ 7] = (riff_size >> 24) & 0xff;
    memcpy(hdr +  8, "WAVE", 4);
    memcpy(hdr + 12, "fmt ", 4);
    hdr[16] = 16; hdr[17] = 0; hdr[18] = 0; hdr[19] = 0;
    hdr[20] = fmt_tag & 0xff; hdr[21] = 0;
    hdr[22] = channels & 0xff; hdr[23] = 0;
    hdr[24] =  sr        & 0xff;
    hdr[25] = (sr >>  8) & 0xff;
    hdr[26] = (sr >> 16) & 0xff;
    hdr[27] = (sr >> 24) & 0xff;
    hdr[28] =  byte_rate        & 0xff;
    hdr[29] = (byte_rate >>  8) & 0xff;
    hdr[30] = (byte_rate >> 16) & 0xff;
    hdr[31] = (byte_rate >> 24) & 0xff;
    hdr[32] = block_align & 0xff; hdr[33] = 0;
    hdr[34] = bits & 0xff; hdr[35] = 0;
    memcpy(hdr + 36, "data", 4);
    hdr[40] =  data_bytes        & 0xff;
    hdr[41] = (data_bytes >>  8) & 0xff;
    hdr[42] = (data_bytes >> 16) & 0xff;
    hdr[43] = (data_bytes >> 24) & 0xff;

    if (fwrite(hdr, 1, 44, f) != 44 ||
        fwrite(samples, sizeof(int16_t), nframes, f) != nframes) {
        fprintf(stderr, "wav_write: write failed\n");
        fclose(f); return 0;
    }
    fclose(f);
    return 1;
}

/* ---------- main -------------------------------------------------------- */

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr,
            "Usage: %s <model.tar.gz> <input.wav> <output.wav> [atten_lim_db]\n",
            argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* in_path    = argv[2];
    const char* out_path   = argv[3];
    float atten_lim_db = (argc == 5) ? (float)atof(argv[4]) : 100.0f;

    WavData wav;
    if (!wav_load(in_path, &wav)) return 1;
    fprintf(stderr, "Input: %s (%u Hz, %u ch, %zu frames)\n",
            in_path, wav.sample_rate, wav.channels, wav.nframes);

    /* Downmix stereo to mono in-place. */
    int16_t* mono = (int16_t*)malloc(wav.nframes * sizeof(int16_t));
    if (!mono) { fprintf(stderr, "OOM\n"); free(wav.samples); return 1; }
    if (wav.channels == 1) {
        memcpy(mono, wav.samples, wav.nframes * sizeof(int16_t));
    } else {
        for (size_t i = 0; i < wav.nframes; i++) {
            int32_t l = wav.samples[2 * i];
            int32_t r = wav.samples[2 * i + 1];
            mono[i] = (int16_t)((l + r) / 2);
        }
    }
    free(wav.samples);
    wav.samples = NULL;

    WeyaModel* model = weya_nc_model_load_from_path(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        free(mono); return 1;
    }

    WeyaSession* session = weya_nc_session_create(model, wav.sample_rate, atten_lim_db);
    if (!session) {
        fprintf(stderr, "Failed to create session\n");
        weya_nc_model_free(model);
        free(mono); return 1;
    }

    size_t frame_len = weya_nc_get_frame_length(session);
    float* in_f32  = (float*)calloc(frame_len, sizeof(float));
    float* out_f32 = (float*)calloc(frame_len, sizeof(float));
    int16_t* out_i16 = (int16_t*)calloc(wav.nframes, sizeof(int16_t));
    if (!in_f32 || !out_f32 || !out_i16) {
        fprintf(stderr, "OOM\n");
        free(in_f32); free(out_f32); free(out_i16); free(mono);
        weya_nc_session_free(session); weya_nc_model_free(model);
        return 1;
    }

    size_t idx = 0;
    while (idx < wav.nframes) {
        size_t n = (idx + frame_len <= wav.nframes) ? frame_len : (wav.nframes - idx);
        for (size_t i = 0; i < frame_len; i++) {
            in_f32[i] = (i < n) ? ((float)mono[idx + i] / 32768.0f) : 0.0f;
        }
        (void)weya_nc_process_frame(session, in_f32, out_f32);
        for (size_t i = 0; i < n; i++) {
            float v = clampf(out_f32[i] * 32768.0f, -32768.0f, 32767.0f);
            out_i16[idx + i] = (int16_t)v;
        }
        idx += n;
    }

    int ok = wav_write_mono_i16(out_path, out_i16, wav.nframes, wav.sample_rate);

    free(in_f32); free(out_f32); free(out_i16); free(mono);
    weya_nc_session_free(session);
    weya_nc_model_free(model);

    if (!ok) return 1;
    fprintf(stderr, "Saved: %s\n", out_path);
    return 0;
}
