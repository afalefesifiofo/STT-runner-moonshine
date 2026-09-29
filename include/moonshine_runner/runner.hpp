// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// include/moonshine_runner/runner.hpp
//
// Public C++ API for MoonshineRunner — a lightweight, cross-platform
// Speech-to-Text library backed by Moonshine ONNX models via ONNX Runtime.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace moonshine {

/// Which Moonshine model variant to load.
enum class ModelSize {
    Tiny,   ///< ~27 MB INT8 — best for edge/mobile
    Base,   ///< ~60 MB INT8 — higher accuracy
};

/// Transcription result returned by MoonshineRunner::transcribe().
struct TranscriptResult {
    std::string text;           ///< Decoded text
    double      rtf  = 0.0;    ///< Real-Time Factor  (inference_ms / audio_ms)
    double      ttft_ms = 0.0; ///< Time to First Token in milliseconds
};

/// Configuration options for MoonshineRunner.
struct RunnerConfig {
    std::string model_dir;                ///< Path to directory containing *.onnx and tokenizer.json
    ModelSize   model_size = ModelSize::Tiny;
    int         intra_op_threads = 4;    ///< ORT intra-op parallelism
    int         inter_op_threads = 1;    ///< ORT inter-op parallelism
    bool        use_arm_xnnpack  = false;///< Enable XNNPACK EP (Arm targets)
    int         max_tokens       = 448;  ///< Maximum decoder steps
};

// Forward declaration of implementation class (PIMPL)
class RunnerImpl;

/// ---------------------------------------------------------------------------
/// MoonshineRunner
///
/// Thread-safety: a single MoonshineRunner instance must not be called from
/// multiple threads simultaneously. Create one instance per thread.
/// ---------------------------------------------------------------------------
class MoonshineRunner {
public:
    /// Construct and load ONNX sessions.
    /// Throws std::runtime_error if model files are missing or ORT init fails.
    explicit MoonshineRunner(const RunnerConfig& config);

    ~MoonshineRunner();

    // Non-copyable, movable
    MoonshineRunner(const MoonshineRunner&)            = delete;
    MoonshineRunner& operator=(const MoonshineRunner&) = delete;
    MoonshineRunner(MoonshineRunner&&)                 = default;
    MoonshineRunner& operator=(MoonshineRunner&&)      = default;

    /// Transcribe a complete audio segment.
    ///
    /// @param pcm       Raw float32 PCM samples, 16 kHz, mono, range [-1, 1].
    /// @param n_samples Number of samples (i.e. audio_seconds * 16000).
    /// @returns         TranscriptResult with decoded text, RTF, and TTFT.
    TranscriptResult transcribe(const float* pcm, size_t n_samples);

    /// Convenience overload accepting a std::vector.
    TranscriptResult transcribe(const std::vector<float>& pcm) {
        return transcribe(pcm.data(), pcm.size());
    }

    /// Streaming interface: push one chunk at a time.
    /// Call begin_stream() before the first chunk, push_chunk() for each
    /// audio segment, and end_stream() to flush and get the final transcript.
    void begin_stream();
    void push_chunk(const float* pcm, size_t n_samples);
    TranscriptResult end_stream();

    /// Return the sample rate expected by the model (always 16000 Hz).
    static constexpr int sample_rate() { return 16000; }

    /// ORT provider string (informational).
    std::string execution_provider() const;

private:
    std::unique_ptr<RunnerImpl> impl_;
};

} // namespace moonshine
