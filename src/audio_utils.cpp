// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// src/audio_utils.cpp

#include "moonshine_runner/audio_utils.hpp"

#include <sndfile.h>
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace moonshine {
namespace audio {

// ---------------------------------------------------------------------------
// Linear resampler — adequate for small rate differences (e.g. 44100 → 16000).
// For production quality, prefer libsamplerate.
// ---------------------------------------------------------------------------
std::vector<float> resample_linear(
    const float* in, size_t n_in,
    int src_rate, int dst_rate)
{
    if (src_rate == dst_rate)
        return std::vector<float>(in, in + n_in);

    double ratio = static_cast<double>(dst_rate) / src_rate;
    size_t n_out = static_cast<size_t>(n_in * ratio);
    std::vector<float> out(n_out);

    for (size_t i = 0; i < n_out; ++i) {
        double pos = i / ratio;
        size_t lo  = static_cast<size_t>(pos);
        size_t hi  = lo + 1;
        double frac = pos - lo;
        float v_lo = in[lo];
        float v_hi = (hi < n_in) ? in[hi] : v_lo;
        out[i] = static_cast<float>(v_lo + frac * (v_hi - v_lo));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Load any audio file supported by libsndfile, resample to 16 kHz mono.
// ---------------------------------------------------------------------------
std::vector<float> load_audio_file(const std::string& path) {
    SF_INFO info{};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf)
        throw std::runtime_error(std::string("Cannot open audio: ") + sf_strerror(nullptr)
                                 + " (" + path + ")");

    size_t n_frames  = static_cast<size_t>(info.frames);
    int    channels  = info.channels;
    int    src_rate  = info.samplerate;

    std::vector<float> raw(n_frames * channels);
    sf_count_t read = sf_readf_float(sf, raw.data(), info.frames);
    sf_close(sf);

    if (read != info.frames)
        throw std::runtime_error("Incomplete read: " + path);

    // Mix down to mono
    std::vector<float> mono(n_frames);
    if (channels == 1) {
        mono = std::move(raw);
    } else {
        for (size_t i = 0; i < n_frames; ++i) {
            float sum = 0.f;
            for (int c = 0; c < channels; ++c)
                sum += raw[i * channels + c];
            mono[i] = sum / channels;
        }
    }

    // Resample to 16 kHz
    return resample_linear(mono.data(), mono.size(), src_rate, 16000);
}

// ---------------------------------------------------------------------------
// Read raw float32 LE from stdin until EOF.
// ---------------------------------------------------------------------------
std::vector<float> read_raw_stdin() {
    std::vector<float> out;
    float sample;
    while (std::fread(&sample, sizeof(float), 1, stdin) == 1)
        out.push_back(sample);
    return out;
}

// ---------------------------------------------------------------------------
// int16 → float32
// ---------------------------------------------------------------------------
std::vector<float> int16_to_float(const int16_t* in, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = in[i] / 32768.0f;
    return out;
}

} // namespace audio
} // namespace moonshine
