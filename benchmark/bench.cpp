// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// benchmark/bench.cpp
//
// Measures RTF (Real-Time Factor), TTFT (Time to First Token), and
// peak RSS for a given WAV file, across N runs (warm + cold stats).
//
// Usage:
//   bench [--runs N] [--model-dir PATH] [--model-size tiny|base] <wav_file>

#include "moonshine_runner/runner.hpp"
#include "moonshine_runner/audio_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef __linux__
#include <sys/resource.h>
static long peak_rss_kb() {
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}
#else
static long peak_rss_kb() { return 0; }
#endif

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: bench [--runs N] [--model-dir PATH] "
                        "[--model-size tiny|base] <wav_file>\n");
        return 1;
    }

    moonshine::RunnerConfig cfg;
    const char* env_dir = std::getenv("MOONSHINE_MODEL_DIR");
    cfg.model_dir = env_dir ? env_dir : "models/moonshine-tiny";

    int runs = 5;
    std::string wav_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--runs" && i+1 < argc)
            runs = std::stoi(argv[++i]);
        else if (arg == "--model-dir" && i+1 < argc)
            cfg.model_dir = argv[++i];
        else if (arg == "--model-size" && i+1 < argc) {
            std::string sz(argv[++i]);
            cfg.model_size = (sz == "base") ? moonshine::ModelSize::Base
                                            : moonshine::ModelSize::Tiny;
        } else if (arg == "--threads" && i+1 < argc)
            cfg.intra_op_threads = std::stoi(argv[++i]);
        else if (arg == "--arm-xnnpack")
            cfg.use_arm_xnnpack = true;
        else if (!arg.empty() && arg[0] != '-')
            wav_path = arg;
    }

    if (wav_path.empty()) {
        fprintf(stderr, "Error: no WAV file specified.\n");
        return 1;
    }

    try {
        auto pcm = moonshine::audio::load_audio_file(wav_path);
        double audio_dur_ms = static_cast<double>(pcm.size()) / 16000.0 * 1000.0;

        fprintf(stderr, "Loading model from: %s\n", cfg.model_dir.c_str());
        moonshine::MoonshineRunner runner(cfg);
        fprintf(stderr, "EP: %s\n", runner.execution_provider().c_str());
        fprintf(stderr, "Audio: %.2f s  (%zu samples)\n",
                audio_dur_ms / 1000.0, pcm.size());
        fprintf(stderr, "Runs: %d\n\n", runs);

        // Warm-up run (not counted)
        auto warm = runner.transcribe(pcm);
        fprintf(stderr, "Transcript: \"%s\"\n\n", warm.text.c_str());

        std::vector<double> rtf_vals, ttft_vals;
        for (int r = 0; r < runs; ++r) {
            auto res = runner.transcribe(pcm);
            rtf_vals.push_back(res.rtf);
            ttft_vals.push_back(res.ttft_ms);
            fprintf(stderr, "  run %d: RTF=%.3f  TTFT=%.1f ms\n",
                    r + 1, res.rtf, res.ttft_ms);
        }

        // Statistics
        auto stats = [](const std::vector<double>& v) {
            double sum  = std::accumulate(v.begin(), v.end(), 0.0);
            double mean = sum / v.size();
            double mn   = *std::min_element(v.begin(), v.end());
            double mx   = *std::max_element(v.begin(), v.end());
            return std::make_tuple(mean, mn, mx);
        };

        auto [rtf_mean, rtf_min, rtf_max]    = stats(rtf_vals);
        auto [ttft_mean, ttft_min, ttft_max] = stats(ttft_vals);
        long rss = peak_rss_kb();

        printf("\n");
        printf("=== Benchmark Results ===\n");
        printf("Audio duration : %.2f s\n", audio_dur_ms / 1000.0);
        printf("Runs           : %d\n", runs);
        printf("RTF  mean/min/max : %.3f / %.3f / %.3f\n", rtf_mean, rtf_min, rtf_max);
        printf("TTFT mean/min/max : %.1f / %.1f / %.1f ms\n", ttft_mean, ttft_min, ttft_max);
        if (rss > 0)
            printf("Peak RSS       : %.1f MB\n", rss / 1024.0);
        printf("EP             : %s\n", runner.execution_provider().c_str());

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
