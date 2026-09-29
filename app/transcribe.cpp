// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// app/transcribe.cpp
//
// CLI: transcribe one or more audio files using Moonshine.
//
// Usage:
//   transcribe [options] <audio_file> [<audio_file> ...]
//
// Options:
//   --model-dir <path>   Directory with ONNX models (default: auto-detect)
//   --model-size <s>     tiny | base  (default: tiny)
//   --threads <n>        Intra-op threads (default: 4)
//   --arm-xnnpack        Enable Arm XNNPACK execution provider
//   --rtf                Print Real-Time Factor
//   --no-text            Suppress transcript (useful with --rtf for benchmarking)

#include "moonshine_runner/runner.hpp"
#include "moonshine_runner/audio_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>

static void usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s [options] <audio_file> ...\n"
        "Options:\n"
        "  --model-dir <path>   ONNX model directory\n"
        "  --model-size <s>     tiny|base  (default: tiny)\n"
        "  --threads <n>        CPU threads (default: 4)\n"
        "  --arm-xnnpack        Enable Arm XNNPACK EP\n"
        "  --rtf                Show Real-Time Factor\n"
        "  --no-text            Suppress transcript output\n",
        argv0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    moonshine::RunnerConfig cfg;
    bool show_rtf   = false;
    bool show_text  = true;
    std::vector<std::string> files;

    // Default model dir: try env var, then sibling "models/" dir
    const char* env_dir = std::getenv("MOONSHINE_MODEL_DIR");
    cfg.model_dir = env_dir ? env_dir : "models/moonshine-tiny";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--model-dir" && i + 1 < argc) {
            cfg.model_dir = argv[++i];
        } else if (arg == "--model-size" && i + 1 < argc) {
            std::string sz(argv[++i]);
            cfg.model_size = (sz == "base") ? moonshine::ModelSize::Base
                                            : moonshine::ModelSize::Tiny;
            if (sz == "base") cfg.model_dir = "models/moonshine-base";
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.intra_op_threads = std::stoi(argv[++i]);
        } else if (arg == "--arm-xnnpack") {
            cfg.use_arm_xnnpack = true;
        } else if (arg == "--rtf") {
            show_rtf = true;
        } else if (arg == "--no-text") {
            show_text = false;
        } else if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        } else {
            files.push_back(arg);
        }
    }

    if (files.empty()) {
        fprintf(stderr, "Error: no input files specified.\n");
        usage(argv[0]);
        return 1;
    }

    try {
        moonshine::MoonshineRunner runner(cfg);
        fprintf(stderr, "MoonshineRunner ready (EP: %s)\n",
                runner.execution_provider().c_str());

        for (const auto& path : files) {
            try {
                auto pcm = moonshine::audio::load_audio_file(path);
                auto res = runner.transcribe(pcm);

                if (show_text) {
                    printf("%s\n", res.text.c_str());
                }
                if (show_rtf) {
                    fprintf(stderr, "[%s] RTF=%.3f  TTFT=%.1f ms\n",
                            path.c_str(), res.rtf, res.ttft_ms);
                }
            } catch (const std::exception& e) {
                fprintf(stderr, "Error processing %s: %s\n", path.c_str(), e.what());
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Runner init failed: %s\n", e.what());
        return 1;
    }

    return 0;
}
