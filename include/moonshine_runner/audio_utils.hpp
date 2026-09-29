// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// include/moonshine_runner/audio_utils.hpp
//
// Lightweight audio helpers: resampling and basic WAV reading.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace moonshine {
namespace audio {

/// Resample PCM from src_rate to dst_rate (linear interpolation).
/// For production use, prefer libsamplerate (SRC_SINC_BEST_QUALITY).
std::vector<float> resample_linear(
    const float* in, size_t n_in,
    int src_rate, int dst_rate);

/// Load a WAV/FLAC/OGG file and return 16 kHz mono float32 PCM.
/// Uses libsndfile. Throws std::runtime_error on failure.
std::vector<float> load_audio_file(const std::string& path);

/// Read from stdin (raw float32 LE, 16 kHz mono).
/// Reads until EOF. Useful for piping: `arecord -r 16000 -f FLOAT_LE | ./live`
std::vector<float> read_raw_stdin();

/// Clamp samples to [-1, 1] and convert int16 → float32.
std::vector<float> int16_to_float(const int16_t* in, size_t n);

} // namespace audio
} // namespace moonshine
