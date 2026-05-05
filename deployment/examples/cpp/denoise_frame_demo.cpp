// Minimal C++17 example: denoise a WAV file with libweya_nc.
//
// Build (macOS / Linux):
//   c++ -std=c++17 -O2 -Wall -Wextra denoise_frame_demo.cpp \
//       -I../../include -L../../lib -lweya_nc \
//       -Wl,-rpath,@loader_path/../../lib   # macOS
//       # -Wl,-rpath,'$ORIGIN/../../lib'    # Linux
//       -o denoise_frame_demo_cpp
//
// Usage:
//   ./denoise_frame_demo_cpp <model.tar.gz> <input.wav> <output.wav> [atten_lim_db]
//
// Accepts 16-bit PCM WAV, mono or stereo (auto-downmixed). Sample rate
// read from the WAV header.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include "../../include/weya_nc.h"
}

namespace {

struct WavData {
    uint32_t sample_rate    = 0;
    uint16_t channels       = 0;
    uint16_t bits_per_sample = 0;
    std::vector<int16_t> samples;  // interleaved
    size_t nframes() const { return channels ? samples.size() / channels : 0; }
};

uint16_t rd_u16(std::istream& s) {
    unsigned char b[2];
    s.read(reinterpret_cast<char*>(b), 2);
    return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}
uint32_t rd_u32(std::istream& s) {
    unsigned char b[4];
    s.read(reinterpret_cast<char*>(b), 4);
    return static_cast<uint32_t>(b[0])
         | (static_cast<uint32_t>(b[1]) << 8)
         | (static_cast<uint32_t>(b[2]) << 16)
         | (static_cast<uint32_t>(b[3]) << 24);
}

WavData wav_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    char id[4];
    f.read(id, 4);
    rd_u32(f);  // riff size
    char wave[4];
    f.read(wave, 4);
    if (std::memcmp(id, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0)
        throw std::runtime_error("not a RIFF/WAVE file");

    WavData w;
    bool got_fmt = false;
    uint16_t fmt_tag = 0;

    while (f) {
        char cid[4];
        f.read(cid, 4);
        if (f.gcount() != 4) throw std::runtime_error("EOF in chunk header");
        uint32_t sz = rd_u32(f);

        if (std::memcmp(cid, "fmt ", 4) == 0) {
            std::streampos start = f.tellg();
            fmt_tag         = rd_u16(f);
            w.channels      = rd_u16(f);
            w.sample_rate   = rd_u32(f);
            rd_u32(f);  // byte rate
            rd_u16(f);  // block align
            w.bits_per_sample = rd_u16(f);
            std::streamoff consumed = f.tellg() - start;
            if (static_cast<std::streamoff>(sz) > consumed)
                f.seekg(sz - consumed, std::ios::cur);
            if (sz & 1) f.seekg(1, std::ios::cur);
            got_fmt = true;
        } else if (std::memcmp(cid, "data", 4) == 0) {
            if (!got_fmt) throw std::runtime_error("data before fmt");
            if (fmt_tag != 1) throw std::runtime_error("only PCM supported");
            if (w.bits_per_sample != 16) throw std::runtime_error("only 16-bit PCM supported");
            if (w.channels < 1 || w.channels > 2) throw std::runtime_error("only mono/stereo supported");
            size_t total = sz / sizeof(int16_t);
            w.samples.resize(total);
            f.read(reinterpret_cast<char*>(w.samples.data()), sz);
            if (static_cast<size_t>(f.gcount()) != sz)
                throw std::runtime_error("truncated data chunk");
            return w;
        } else {
            f.seekg(sz + (sz & 1), std::ios::cur);
        }
    }
    throw std::runtime_error("no data chunk");
}

void wav_write_mono_i16(const std::string& path,
                        const std::vector<int16_t>& samples, uint32_t sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    const uint32_t data_bytes  = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint16_t channels    = 1, bits = 16, block_align = 2, fmt_tag = 1;
    const uint32_t byte_rate   = sr * channels * (bits / 8);
    const uint32_t riff_size   = 4 + (8 + 16) + (8 + data_bytes);

    auto wr_u32 = [&](uint32_t v) {
        unsigned char b[4] = {
            static_cast<unsigned char>(v & 0xff),
            static_cast<unsigned char>((v >> 8) & 0xff),
            static_cast<unsigned char>((v >> 16) & 0xff),
            static_cast<unsigned char>((v >> 24) & 0xff),
        };
        f.write(reinterpret_cast<const char*>(b), 4);
    };
    auto wr_u16 = [&](uint16_t v) {
        unsigned char b[2] = {
            static_cast<unsigned char>(v & 0xff),
            static_cast<unsigned char>((v >> 8) & 0xff),
        };
        f.write(reinterpret_cast<const char*>(b), 2);
    };

    f.write("RIFF", 4); wr_u32(riff_size); f.write("WAVE", 4);
    f.write("fmt ", 4); wr_u32(16);
    wr_u16(fmt_tag); wr_u16(channels);
    wr_u32(sr); wr_u32(byte_rate);
    wr_u16(block_align); wr_u16(bits);
    f.write("data", 4); wr_u32(data_bytes);
    f.write(reinterpret_cast<const char*>(samples.data()), data_bytes);
}

// RAII handles for the C library.
struct ModelDeleter   { void operator()(WeyaModel* m)   const noexcept { if (m) weya_nc_model_free(m); } };
struct SessionDeleter { void operator()(WeyaSession* s) const noexcept { if (s) weya_nc_session_free(s); } };
using ModelPtr   = std::unique_ptr<WeyaModel,   ModelDeleter>;
using SessionPtr = std::unique_ptr<WeyaSession, SessionDeleter>;

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
            "Usage: %s <model.tar.gz> <input.wav> <output.wav> [atten_lim_db]\n",
            argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string in_path    = argv[2];
    const std::string out_path   = argv[3];
    const float atten_lim_db = (argc == 5) ? std::stof(argv[4]) : 100.0f;

    WavData wav = wav_load(in_path);
    std::fprintf(stderr, "Input: %s (%u Hz, %u ch, %zu frames)\n",
                 in_path.c_str(), wav.sample_rate, wav.channels, wav.nframes());

    // Downmix to mono.
    std::vector<int16_t> mono(wav.nframes());
    if (wav.channels == 1) {
        std::copy(wav.samples.begin(), wav.samples.end(), mono.begin());
    } else {
        for (size_t i = 0; i < mono.size(); ++i) {
            int32_t l = wav.samples[2 * i];
            int32_t r = wav.samples[2 * i + 1];
            mono[i] = static_cast<int16_t>((l + r) / 2);
        }
    }

    ModelPtr model{ weya_nc_model_load_from_path(model_path.c_str()) };
    if (!model) throw std::runtime_error("failed to load model: " + model_path);

    SessionPtr session{ weya_nc_session_create(model.get(), wav.sample_rate, atten_lim_db) };
    if (!session) throw std::runtime_error("failed to create session");

    const size_t frame_len = weya_nc_get_frame_length(session.get());
    std::vector<float>   in_f32(frame_len, 0.0f);
    std::vector<float>   out_f32(frame_len, 0.0f);
    std::vector<int16_t> out_i16(mono.size(), 0);

    for (size_t idx = 0; idx < mono.size(); ) {
        const size_t n = std::min(frame_len, mono.size() - idx);
        for (size_t i = 0; i < frame_len; ++i) {
            in_f32[i] = (i < n) ? static_cast<float>(mono[idx + i]) / 32768.0f : 0.0f;
        }
        weya_nc_process_frame(session.get(), in_f32.data(), out_f32.data());
        for (size_t i = 0; i < n; ++i) {
            float v = std::clamp(out_f32[i] * 32768.0f, -32768.0f, 32767.0f);
            out_i16[idx + i] = static_cast<int16_t>(v);
        }
        idx += n;
    }

    wav_write_mono_i16(out_path, out_i16, wav.sample_rate);
    std::fprintf(stderr, "Saved: %s\n", out_path.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
}
