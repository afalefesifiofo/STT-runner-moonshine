// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// app/live.cpp
//
// CLI: real-time live transcription from microphone.
//
// Reads audio from PortAudio (if HAVE_PORTAUDIO=1) or from stdin
// (pipe: `arecord -r 16000 -c 1 -f FLOAT_LE -q | ./live`).
//
// Uses a sliding 2-second window with 0.5-second hop (simple VAD-free
// approach). For production, consider Silero VAD.

#include "moonshine_runner/runner.hpp"
#include "moonshine_runner/audio_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <atomic>
#include <vector>
#include <string>
#include <deque>
#include <stdexcept>

#ifdef HAVE_PORTAUDIO
#  include <portaudio.h>
#endif

static constexpr int SAMPLE_RATE    = 16000;
static constexpr int WINDOW_SECS    = 2;
static constexpr int HOP_SECS       = 1;
static constexpr int WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECS;
static constexpr int HOP_SAMPLES    = SAMPLE_RATE * HOP_SECS;

static std::atomic<bool> g_running{true};
static void sig_handler(int) { g_running = false; }

// ---------------------------------------------------------------------------
// Stdin mode: read chunks from piped raw float32 PCM
// ---------------------------------------------------------------------------
static int run_stdin(moonshine::MoonshineRunner& runner) {
    fprintf(stderr, "Live mode (stdin) — pipe 16 kHz mono float32 PCM.\n");
    fprintf(stderr, "Example: arecord -r 16000 -c 1 -f FLOAT_LE -q | ./live\n");
    fprintf(stderr, "Press Ctrl+C to stop.\n\n");

    std::deque<float> ring;

    float buf[HOP_SAMPLES];
    while (g_running) {
        size_t got = std::fread(buf, sizeof(float), HOP_SAMPLES, stdin);
        if (got == 0) break;

        ring.insert(ring.end(), buf, buf + got);

        if (static_cast<int>(ring.size()) >= WINDOW_SAMPLES) {
            std::vector<float> window(ring.begin(), ring.begin() + WINDOW_SAMPLES);
            auto res = runner.transcribe(window);
            if (!res.text.empty())
                printf("\r\033[K%s", res.text.c_str());
            fflush(stdout);

            // Slide: remove one hop worth of samples
            for (int i = 0; i < HOP_SAMPLES; ++i)
                ring.pop_front();
        }
    }
    printf("\n");
    return 0;
}

// ---------------------------------------------------------------------------
// PortAudio mode
// ---------------------------------------------------------------------------
#ifdef HAVE_PORTAUDIO

struct PaUserData {
    std::deque<float>* ring;
};

static int pa_callback(
    const void* in, void*, unsigned long frames,
    const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* user_data)
{
    auto* ud = static_cast<PaUserData*>(user_data);
    const float* pcm = static_cast<const float*>(in);
    ud->ring->insert(ud->ring->end(), pcm, pcm + frames);
    return g_running ? paContinue : paComplete;
}

static int run_portaudio(moonshine::MoonshineRunner& runner) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "PortAudio init error: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    std::deque<float> ring;
    PaUserData ud{&ring};

    PaStream* stream;
    err = Pa_OpenDefaultStream(&stream,
        1, 0,         // 1 input, 0 output channels
        paFloat32,    // sample format
        SAMPLE_RATE,
        HOP_SAMPLES,  // frames per buffer (one hop)
        pa_callback, &ud);

    if (err != paNoError) {
        fprintf(stderr, "Pa_OpenDefaultStream error: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        return 1;
    }

    Pa_StartStream(stream);
    fprintf(stderr, "Live mode (PortAudio) — speak into your microphone.\n");
    fprintf(stderr, "Press Ctrl+C to stop.\n\n");

    while (g_running) {
        Pa_Sleep(100); // 100 ms poll
        if (static_cast<int>(ring.size()) >= WINDOW_SAMPLES) {
            std::vector<float> window(ring.begin(), ring.begin() + WINDOW_SAMPLES);
            auto res = runner.transcribe(window);
            if (!res.text.empty())
                printf("\r\033[K%s", res.text.c_str());
            fflush(stdout);
            for (int i = 0; i < HOP_SAMPLES; ++i) ring.pop_front();
        }
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    printf("\n");
    return 0;
}
#endif // HAVE_PORTAUDIO

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    moonshine::RunnerConfig cfg;
    const char* env_dir = std::getenv("MOONSHINE_MODEL_DIR");
    cfg.model_dir = env_dir ? env_dir : "models/moonshine-tiny";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--model-dir" && i+1 < argc)
            cfg.model_dir = argv[++i];
        else if (arg == "--model-size" && i+1 < argc) {
            std::string sz(argv[++i]);
            cfg.model_size = (sz == "base") ? moonshine::ModelSize::Base
                                            : moonshine::ModelSize::Tiny;
        } else if (arg == "--threads" && i+1 < argc)
            cfg.intra_op_threads = std::stoi(argv[++i]);
        else if (arg == "--arm-xnnpack")
            cfg.use_arm_xnnpack = true;
    }

    try {
        moonshine::MoonshineRunner runner(cfg);
        fprintf(stderr, "MoonshineRunner ready (EP: %s)\n",
                runner.execution_provider().c_str());

#ifdef HAVE_PORTAUDIO
        return run_portaudio(runner);
#else
        return run_stdin(runner);
#endif
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
