// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// tests/test_audio_utils.cpp

#include "moonshine_runner/audio_utils.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_resample_identity() {
    std::vector<float> in = {0.1f, 0.2f, 0.3f, 0.4f};
    auto out = moonshine::audio::resample_linear(in.data(), in.size(), 16000, 16000);
    assert(out.size() == in.size());
    for (size_t i = 0; i < in.size(); ++i)
        assert(std::abs(out[i] - in[i]) < 1e-5f);
    printf("[PASS] resample_identity\n");
}

static void test_resample_downsample() {
    // 44100 → 16000: ratio ≈ 0.363
    std::vector<float> in(44100, 0.5f); // 1 second of constant signal
    auto out = moonshine::audio::resample_linear(in.data(), in.size(), 44100, 16000);
    assert(out.size() == 16000);
    // All samples should be ~0.5 (constant signal)
    for (float v : out)
        assert(std::abs(v - 0.5f) < 1e-4f);
    printf("[PASS] resample_downsample (44100 → 16000)\n");
}

static void test_int16_to_float() {
    int16_t samples[] = {0, 32767, -32768, 16384};
    auto out = moonshine::audio::int16_to_float(samples, 4);
    assert(out.size() == 4);
    assert(std::abs(out[0]) < 1e-5f);
    assert(std::abs(out[1] - (32767.0f / 32768.0f)) < 1e-4f);
    assert(std::abs(out[2] - (-1.0f)) < 1e-4f);
    printf("[PASS] int16_to_float\n");
}

int main() {
    test_resample_identity();
    test_resample_downsample();
    test_int16_to_float();
    printf("All audio_utils tests passed.\n");
    return 0;
}
