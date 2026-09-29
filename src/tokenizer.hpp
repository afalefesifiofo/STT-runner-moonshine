// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// src/tokenizer.hpp  (internal header)
//
// Wraps the Moonshine tokenizer.json (HuggingFace tokenizers format).
// Uses the sentencepiece C library for decoding.
// The tokenizer.json maps token IDs ↔ byte-level BPE pieces.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <fstream>

#include <nlohmann/json.hpp>  // header-only, fetched via FetchContent

namespace moonshine {

/// Minimal JSON-based tokenizer for Moonshine.
/// Decodes token ID sequences → UTF-8 text using the vocab embedded in tokenizer.json.
class MoonshineTokenizer {
public:
    explicit MoonshineTokenizer(const std::string& tokenizer_json_path) {
        std::ifstream f(tokenizer_json_path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open tokenizer: " + tokenizer_json_path);

        nlohmann::json j;
        f >> j;

        // Hugging Face tokenizers format: model.vocab is {token_str: id}
        auto& vocab = j.at("model").at("vocab");
        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            int64_t id = it.value().get<int64_t>();
            id_to_token_[id] = it.key();
        }

        // Also store added_tokens (special tokens like <|endoftext|>)
        if (j.contains("added_tokens")) {
            for (auto& at : j["added_tokens"]) {
                int64_t id  = at["id"].get<int64_t>();
                std::string content = at["content"].get<std::string>();
                id_to_token_[id] = content;
            }
        }
    }

    /// Convert a sequence of token IDs to a UTF-8 string.
    /// Skips special tokens (BOS=1, EOS=2) and handles Ġ (GPT-style space prefix).
    std::string decode(const std::vector<int64_t>& ids) const {
        std::string result;
        result.reserve(ids.size() * 4);
        for (int64_t id : ids) {
            if (id <= 2) continue; // BOS / EOS / PAD
            auto it = id_to_token_.find(id);
            if (it == id_to_token_.end()) continue;
            std::string piece = it->second;
            // GPT-2 byte-level BPE: Ġ (U+0120) → space
            piece = replace_gpt2_space(piece);
            result += piece;
        }
        // Trim leading/trailing whitespace
        size_t s = result.find_first_not_of(' ');
        size_t e = result.find_last_not_of(' ');
        if (s == std::string::npos) return "";
        return result.substr(s, e - s + 1);
    }

private:
    std::unordered_map<int64_t, std::string> id_to_token_;

    static std::string replace_gpt2_space(const std::string& s) {
        // Ġ is UTF-8: 0xC4 0xA0
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c == 0xC4 && i + 1 < s.size() &&
                static_cast<unsigned char>(s[i+1]) == 0xA0) {
                out += ' ';
                i += 2;
            } else {
                out += s[i++];
            }
        }
        return out;
    }
};

} // namespace moonshine
