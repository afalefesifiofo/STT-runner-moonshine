// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// src/runner.cpp
//
// MoonshineRunner implementation.
//
// Moonshine model I/O — onnx-community/moonshine-{tiny,base}-ONNX:
//
//   encoder_model_int8.onnx
//     Input:  input_values [batch=1, n_samples]  (float32)
//     Output: last_hidden_state [1, enc_seq, 288]
//
//   decoder_model_merged_int8.onnx
//     Inputs:
//       input_ids                        [1, 1]         int64
//       encoder_hidden_states            [1, enc_seq, 288] float32
//       use_cache_branch                 [1]            bool
//       past_key_values.L.decoder.key    [1, 8, past, 36] float32  (L=0..5)
//       past_key_values.L.decoder.value  [1, 8, past, 36] float32
//       past_key_values.L.encoder.key    [1, 8, enc_seq_out, 36] float32
//       past_key_values.L.encoder.value  [1, 8, enc_seq_out, 36] float32
//     Outputs:
//       logits                           [1, 1, 32768]
//       present.L.decoder.{key,value}    (updated decoder KV)
//       present.L.encoder.{key,value}    (encoder cross-attn KV, fixed after step 0)

#include "moonshine_runner/runner.hpp"
#include "moonshine_runner/audio_utils.hpp"
#include "tokenizer.hpp"   // internal header

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace moonshine {

// ---------------------------------------------------------------------------
// Constants (Moonshine tiny/base shared)
// ---------------------------------------------------------------------------
static constexpr int64_t BOS_TOKEN = 1;
static constexpr int64_t EOS_TOKEN = 2;
static constexpr int     SAMPLE_RATE = 16000;

// ---------------------------------------------------------------------------
// RunnerImpl
// ---------------------------------------------------------------------------
class RunnerImpl {
public:
    explicit RunnerImpl(const RunnerConfig& cfg)
        : config_(cfg)
        , env_(ORT_LOGGING_LEVEL_WARNING, "MoonshineRunner")
    {
        namespace fs = std::filesystem;

        // Build model paths
        std::string size_str = (cfg.model_size == ModelSize::Tiny) ? "tiny" : "base";
        fs::path dir(cfg.model_dir);
        if (!fs::exists(dir)) {
            // Try the conventional layout next to the binary
            dir = fs::path("models") / ("moonshine-" + size_str);
        }

        encoder_path_  = (dir / "encoder_model_int8.onnx").string();
        decoder_path_  = (dir / "decoder_model_merged_int8.onnx").string();
        tokenizer_path_= (dir / "tokenizer.json").string();

        if (!fs::exists(encoder_path_))
            throw std::runtime_error("Encoder model not found: " + encoder_path_);
        if (!fs::exists(decoder_path_))
            throw std::runtime_error("Decoder model not found: " + decoder_path_);
        if (!fs::exists(tokenizer_path_))
            throw std::runtime_error("Tokenizer not found: " + tokenizer_path_);

        // Session options
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(cfg.intra_op_threads);
        so.SetInterOpNumThreads(cfg.inter_op_threads);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef MOONSHINE_ARM_XNNPACK
        if (cfg.use_arm_xnnpack) {
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_XNNPACK(so, nullptr);
            if (status) {
                const char* msg = Ort::GetApi().GetErrorMessage(status);
                fprintf(stderr, "[MoonshineRunner] XNNPACK EP warning: %s\n", msg);
                Ort::GetApi().ReleaseStatus(status);
            } else {
                ep_name_ = "XNNPACK";
            }
        }
#endif
        if (ep_name_.empty()) ep_name_ = "CPU";

        // Load sessions
        encoder_session_ = std::make_unique<Ort::Session>(env_, encoder_path_.c_str(), so);
        decoder_session_ = std::make_unique<Ort::Session>(env_, decoder_path_.c_str(), so);

        // Load tokenizer
        tokenizer_ = std::make_unique<MoonshineTokenizer>(tokenizer_path_);

        // Inspect decoder to know number of KV cache layers
        size_t n_outputs = decoder_session_->GetOutputCount();
        // Outputs: logits + present.N.decoder.key, present.N.decoder.value, ...
        // Count "present." outputs
        n_kv_layers_ = 0;
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 1; i < n_outputs; ++i) {
            auto name = decoder_session_->GetOutputNameAllocated(i, alloc);
            std::string s(name.get());
            if (s.find("present.") != std::string::npos &&
                s.find(".key") != std::string::npos) {
                ++n_kv_layers_;
            }
        }
    }

    // -----------------------------------------------------------------------
    TranscriptResult transcribe(const float* pcm, size_t n_samples) {
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();

        // 1. Run encoder
        auto enc_out = run_encoder(pcm, n_samples);

        // 2. Decode
        auto t1 = clock::now();
        std::vector<int64_t> token_ids = decode(enc_out, config_.max_tokens, t1);

        auto t2 = clock::now();

        // 3. Decode text
        std::string text = tokenizer_->decode(token_ids);

        // Metrics
        double audio_ms = static_cast<double>(n_samples) / SAMPLE_RATE * 1000.0;
        double infer_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
        double ttft_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();

        TranscriptResult res;
        res.text    = text;
        res.rtf     = infer_ms / audio_ms;
        res.ttft_ms = ttft_ms;
        return res;
    }

    std::string execution_provider() const { return ep_name_; }

    // -----------------------------------------------------------------------
    // Streaming helpers (stateful)
    // -----------------------------------------------------------------------
    void begin_stream() {
        stream_buffer_.clear();
    }

    void push_chunk(const float* pcm, size_t n_samples) {
        stream_buffer_.insert(stream_buffer_.end(), pcm, pcm + n_samples);
    }

    TranscriptResult end_stream() {
        auto result = transcribe(stream_buffer_.data(), stream_buffer_.size());
        stream_buffer_.clear();
        return result;
    }

private:
    // -----------------------------------------------------------------------
    // Run the encoder and return hidden states as a flat buffer + shape
    // -----------------------------------------------------------------------
    struct EncoderOut {
        std::vector<float>   data;
        std::vector<int64_t> shape; // [1, seq, hidden]
    };

    EncoderOut run_encoder(const float* pcm, size_t n_samples) {
        Ort::AllocatorWithDefaultOptions alloc;
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Input tensor: [1, n_samples]
        std::vector<int64_t> audio_shape = {1, static_cast<int64_t>(n_samples)};
        Ort::Value audio_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(pcm), n_samples,
            audio_shape.data(), audio_shape.size());

        const char* input_names[]  = {"audio_signal"};
        const char* output_names[] = {"encoder_hidden_states"};

        auto outputs = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &audio_tensor, 1,
            output_names, 1);

        auto& tensor = outputs[0];
        auto ti = tensor.GetTensorTypeAndShapeInfo();
        auto shape = ti.GetShape();
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);

        float* data_ptr = tensor.GetTensorMutableData<float>();
        return EncoderOut{
            std::vector<float>(data_ptr, data_ptr + n),
            shape
        };
    }

    // -----------------------------------------------------------------------
    // Autoregressive greedy decode with KV-cache
    // -----------------------------------------------------------------------
    std::vector<int64_t> decode(
        const EncoderOut& enc,
        int max_tokens,
        std::chrono::high_resolution_clock::time_point& ttft_out)
    {
        using clock = std::chrono::high_resolution_clock;
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions alloc;

        std::vector<int64_t> generated_ids;
        generated_ids.reserve(max_tokens);

        // Encoder hidden states tensor (reused each step)
        Ort::Value enc_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(enc.data.data()), enc.data.size(),
            enc.shape.data(), enc.shape.size());

        // KV cache: starts empty, grows each step
        // Each layer has 2 tensors (key + value) for decoder self-attention.
        // Shape: [1, n_heads, past_len, head_dim] — starts as [1, n_heads, 0, head_dim]
        // We'll discover n_heads and head_dim from the first decoder output.

        std::vector<std::vector<float>> kv_data(n_kv_layers_ * 2);
        std::vector<std::vector<int64_t>> kv_shapes(n_kv_layers_ * 2);
        int64_t past_len = 0;
        bool first_step = true;

        int64_t cur_token = BOS_TOKEN;

        // Collect input/output name strings from the session
        size_t n_inputs  = decoder_session_->GetInputCount();
        size_t n_outputs = decoder_session_->GetOutputCount();

        std::vector<std::string> in_name_strs(n_inputs);
        std::vector<const char*> in_names(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            in_name_strs[i] = std::string(decoder_session_->GetInputNameAllocated(i, alloc).get());
            in_names[i] = in_name_strs[i].c_str();
        }

        std::vector<std::string> out_name_strs(n_outputs);
        std::vector<const char*> out_names(n_outputs);
        for (size_t i = 0; i < n_outputs; ++i) {
            out_name_strs[i] = std::string(decoder_session_->GetOutputNameAllocated(i, alloc).get());
            out_names[i] = out_name_strs[i].c_str();
        }

        for (int step = 0; step < max_tokens; ++step) {
            // Build input tensors for this step
            std::vector<Ort::Value> input_tensors;
            input_tensors.reserve(n_inputs);

            for (size_t i = 0; i < n_inputs; ++i) {
                const std::string& name = in_name_strs[i];

                if (name == "input_ids") {
                    std::vector<int64_t> ids_shape = {1, 1};
                    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                        mem_info, &cur_token, 1, ids_shape.data(), 2));

                } else if (name == "encoder_hidden_states") {
                    // We need to re-create from enc since tensors aren't copyable
                    input_tensors.push_back(Ort::Value::CreateTensor<float>(
                        mem_info,
                        const_cast<float*>(enc.data.data()), enc.data.size(),
                        enc.shape.data(), enc.shape.size()));

                } else if (name == "use_cache_branch") {
                    static int8_t use_cache_val;
                    use_cache_val = first_step ? 0 : 1;
                    std::vector<int64_t> s = {1};
                    input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
                        mem_info, &use_cache_val, 1, s.data(), 1));

                } else {
                    // KV cache input — find which layer
                    // Name format: past_key_values.N.decoder.key / .value
                    int layer = -1;
                    bool is_key = (name.find(".key") != std::string::npos);

                    for (int l = 0; l < static_cast<int>(n_kv_layers_); ++l) {
                        std::string key_name  = "past_key_values." + std::to_string(l) + ".decoder.key";
                        std::string val_name  = "past_key_values." + std::to_string(l) + ".decoder.value";
                        if (name == key_name || name == val_name) { layer = l; break; }
                    }

                    if (layer < 0) {
                        // Unknown input — push empty tensor
                        std::vector<int64_t> empty_shape = {1, 1, 0, 64};
                        std::vector<float>   empty_data;
                        input_tensors.push_back(Ort::Value::CreateTensor<float>(
                            mem_info, empty_data.data(), 0,
                            empty_shape.data(), empty_shape.size()));
                        continue;
                    }

                    int kv_idx = layer * 2 + (is_key ? 0 : 1);
                    auto& data  = kv_data[kv_idx];
                    auto& shape = kv_shapes[kv_idx];

                    if (shape.empty()) {
                        // First step: empty past
                        shape = {1, 4, 0, 64}; // will be updated after first output
                        input_tensors.push_back(Ort::Value::CreateTensor<float>(
                            mem_info, nullptr, 0, shape.data(), shape.size()));
                    } else {
                        input_tensors.push_back(Ort::Value::CreateTensor<float>(
                            mem_info, data.data(), data.size(),
                            shape.data(), shape.size()));
                    }
                }
            }

            // Run decoder
            auto outputs = decoder_session_->Run(
                Ort::RunOptions{nullptr},
                in_names.data(), input_tensors.data(), n_inputs,
                out_names.data(), n_outputs);

            if (first_step) {
                ttft_out = clock::now();
                first_step = false;
            }

            // 0: logits [1, 1, vocab_size]
            auto& logits_tensor = outputs[0];
            auto logits_info = logits_tensor.GetTensorTypeAndShapeInfo();
            int64_t vocab_size = logits_info.GetShape()[2];
            const float* logits_data = logits_tensor.GetTensorData<float>();

            // Greedy: argmax over vocab
            int64_t best_token = static_cast<int64_t>(
                std::max_element(logits_data, logits_data + vocab_size) - logits_data);

            generated_ids.push_back(best_token);
            cur_token = best_token;

            if (best_token == EOS_TOKEN) break;

            // Update KV cache from present.* outputs
            ++past_len;
            for (size_t oi = 1; oi < n_outputs; ++oi) {
                const std::string& oname = out_name_strs[oi];
                bool is_key = (oname.find(".key") != std::string::npos);
                int layer = -1;
                for (int l = 0; l < static_cast<int>(n_kv_layers_); ++l) {
                    std::string k = "present." + std::to_string(l) + ".decoder.key";
                    std::string v = "present." + std::to_string(l) + ".decoder.value";
                    if (oname == k || oname == v) { layer = l; break; }
                }
                if (layer < 0) continue;

                int kv_idx = layer * 2 + (is_key ? 0 : 1);
                auto ti = outputs[oi].GetTensorTypeAndShapeInfo();
                auto shape = ti.GetShape();
                size_t n = 1;
                for (auto d : shape) n *= static_cast<size_t>(d);
                const float* ptr = outputs[oi].GetTensorData<float>();
                kv_data[kv_idx].assign(ptr, ptr + n);
                kv_shapes[kv_idx] = shape;
            }
        }

        return generated_ids;
    }

    // -----------------------------------------------------------------------
    RunnerConfig config_;
    std::string  ep_name_;

    std::string encoder_path_;
    std::string decoder_path_;
    std::string tokenizer_path_;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;

    std::unique_ptr<MoonshineTokenizer> tokenizer_;

    size_t n_kv_layers_ = 0;

    // Streaming state
    std::vector<float> stream_buffer_;
};

// ---------------------------------------------------------------------------
// MoonshineRunner public methods
// ---------------------------------------------------------------------------

MoonshineRunner::MoonshineRunner(const RunnerConfig& config)
    : impl_(std::make_unique<RunnerImpl>(config)) {}

MoonshineRunner::~MoonshineRunner() = default;

TranscriptResult MoonshineRunner::transcribe(const float* pcm, size_t n_samples) {
    return impl_->transcribe(pcm, n_samples);
}

void MoonshineRunner::begin_stream()                              { impl_->begin_stream(); }
void MoonshineRunner::push_chunk(const float* p, size_t n)       { impl_->push_chunk(p, n); }
TranscriptResult MoonshineRunner::end_stream()                   { return impl_->end_stream(); }
std::string MoonshineRunner::execution_provider() const          { return impl_->execution_provider(); }

} // namespace moonshine
