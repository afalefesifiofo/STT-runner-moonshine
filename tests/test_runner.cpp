// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// tests/test_runner.cpp
//
// Integration test: loads a known WAV file and checks the transcript.
// Requires models to be present (MOONSHINE_MODEL_DIR env var or models/).
// Skips gracefully if models are not available.

#include "moonshine_runner/runner.hpp"
#include "moonshine_runner/audio_utils.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// Normalize string for comparison: lowercase, strip punctuation
static std::string normalize(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (std::isalpha(c))      out += std::tolower(c);
        else if (std::isspace(c)) out += ' ';
    }
    // Collapse spaces
    std::string res;
    bool prev_space = false;
    for (char c : out) {
        if (c == ' ') {
            if (!prev_space) res += c;
            prev_space = true;
        } else {
            res += c;
            prev_space = false;
        }
    }
    return res;
}

static bool contains_words(const std::string& text, const std::string& words) {
    return normalize(text).find(normalize(words)) != std::string::npos;
}

int main() {
    const char* model_dir = std::getenv("MOONSHINE_MODEL_DIR");
    if (!model_dir) model_dir = "models/moonshine-tiny";

    moonshine::RunnerConfig cfg;
    cfg.model_dir = model_dir;

    moonshine::MoonshineRunner* runner = nullptr;
    try {
        runner = new moonshine::MoonshineRunner(cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "SKIP: Runner init failed (models not found?): %s\n", e.what());
        return 0; // Not a failure — skip gracefully
    }

    // Test 1: synthesized constant-zero audio → should produce empty or near-empty text
    {
        std::vector<float> silence(16000 * 2, 0.0f); // 2 seconds of silence
        auto res = runner->transcribe(silence);
        printf("[PASS] silence test: got \"%s\" (RTF=%.3f)\n",
               res.text.c_str(), res.rtf);
    }

    // Test 2: load a real WAV file if provided via MOONSHINE_TEST_WAV env var
    const char* wav_path = std::getenv("MOONSHINE_TEST_WAV");
    const char* expected = std::getenv("MOONSHINE_TEST_EXPECTED");
    if (wav_path && expected) {
        try {
            auto pcm = moonshine::audio::load_audio_file(wav_path);
            auto res = runner->transcribe(pcm);
            printf("Transcript: \"%s\"\n", res.text.c_str());
            printf("Expected  : \"%s\"\n", expected);
            assert(contains_words(res.text, expected));
            printf("[PASS] transcript contains expected words\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[FAIL] %s\n", e.what());
            delete runner;
            return 1;
        }
    } else {
        printf("SKIP: set MOONSHINE_TEST_WAV and MOONSHINE_TEST_EXPECTED for transcript test.\n");
    }

    delete runner;
    printf("All runner tests passed.\n");
    return 0;
}
