#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
# SPDX-License-Identifier: Apache-2.0
#
# bindings/python/moonshine_runner.py
#
# Python binding for MoonshineRunner.
#
# This module provides two usage modes:
#
#   Mode A — Pure Python (ONNX Runtime directly):
#     No compiled C++ library needed. Uses onnxruntime-python directly.
#     This is the easiest way to get started and is fully cross-platform.
#
#   Mode B — ctypes wrapper around libmoonshine_runner.so:
#     Requires the C++ library to be built. Lower overhead, same models.
#
# The module auto-selects Mode A if the shared library is not found.

"""
moonshine_runner — Python bindings for STT-Runner-Moonshine.

Quick start (Mode A, pure Python):
    from bindings.python.moonshine_runner import MoonshineRunner
    runner = MoonshineRunner("models/moonshine-tiny")
    text = runner.transcribe_file("audio.wav")
    print(text)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional: try loading the compiled C++ shared library (Mode B)
# ---------------------------------------------------------------------------
_lib = None  # Will be set if the shared library is found

try:
    import ctypes
    _lib_path = Path(__file__).parent.parent.parent / "build" / "native" / "libmoonshine_runner.so"
    if _lib_path.exists():
        _lib = ctypes.CDLL(str(_lib_path))
        # TODO: define ctypes signatures when C API is added
except Exception:
    pass


@dataclass
class TranscriptResult:
    text: str
    rtf: float = 0.0
    ttft_ms: float = 0.0


class MoonshineRunner:
    """
    Python wrapper for Moonshine ONNX inference.

    Uses ONNX Runtime directly — no C++ compilation required.

    Args:
        model_dir:   Path to directory with encoder_model_int8.onnx,
                     decoder_model_merged_int8.onnx, tokenizer.json.
        intra_op_threads: Number of CPU threads for ORT (default: 4).
        use_xnnpack: Enable XNNPACK execution provider (Arm devices).
    """

    SAMPLE_RATE = 16000
    BOS_TOKEN   = 1
    EOS_TOKEN   = 2
    MAX_TOKENS  = 448

    def __init__(
        self,
        model_dir: Union[str, Path] = "models/moonshine-tiny",
        intra_op_threads: int = 4,
        use_xnnpack: bool = False,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )

        model_dir = Path(model_dir)
        enc_path  = model_dir / "encoder_model_int8.onnx"
        dec_path  = model_dir / "decoder_model_merged_int8.onnx"
        tok_path  = model_dir / "tokenizer.json"

        for p in (enc_path, dec_path, tok_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model file not found: {p}\n"
                    f"Run: python scripts/export_onnx.py --model tiny"
                )

        # Session options
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers: List[str] = []
        if use_xnnpack:
            providers.append("XNNPACKExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._encoder = ort.InferenceSession(str(enc_path), so, providers=providers)
        self._decoder = ort.InferenceSession(str(dec_path), so, providers=providers)
        self._ep      = self._encoder.get_providers()[0]

        # Load vocab
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        self._vocab: dict[int, str] = {}
        for token, idx in tok_data["model"]["vocab"].items():
            self._vocab[int(idx)] = token
        if "added_tokens" in tok_data:
            for at in tok_data["added_tokens"]:
                self._vocab[int(at["id"])] = at["content"]

        # Cache decoder input/output names
        self._dec_input_names  = [i.name for i in self._decoder.get_inputs()]
        self._dec_output_names = [o.name for o in self._decoder.get_outputs()]
        # Count decoder self-attention KV layers only (exclude encoder cross-attn)
        self._n_kv_layers      = sum(
            1 for n in self._dec_output_names
            if "present." in n and ".decoder.key" in n
        )

        print(f"[MoonshineRunner] EP={self._ep}, KV layers={self._n_kv_layers}")

    # -----------------------------------------------------------------------
    def transcribe(self, pcm: np.ndarray) -> TranscriptResult:
        """
        Transcribe raw audio.

        Args:
            pcm: float32 numpy array, shape (n_samples,), 16 kHz, mono.

        Returns:
            TranscriptResult with text, rtf, and ttft_ms.
        """
        assert pcm.dtype == np.float32, "PCM must be float32"
        t0 = time.perf_counter()

        audio_dur_ms = len(pcm) / self.SAMPLE_RATE * 1000.0

        # Encode
        enc_out = self._run_encoder(pcm)
        t1 = time.perf_counter()

        # Decode
        token_ids, ttft_s = self._decode(enc_out)
        t2 = time.perf_counter()

        text = self._decode_tokens(token_ids)
        infer_ms = (t2 - t0) * 1000.0

        return TranscriptResult(
            text    = text,
            rtf     = infer_ms / audio_dur_ms,
            ttft_ms = ttft_s * 1000.0,
        )

    def transcribe_file(self, path: Union[str, Path]) -> str:
        """Load an audio file and return the transcript string."""
        pcm = self.load_audio(path)
        return self.transcribe(pcm).text

    def transcribe_pcm_bytes(self, data: bytes, src_rate: int = 16000) -> str:
        """Transcribe raw int16 PCM bytes (e.g. from PortAudio / sounddevice)."""
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if src_rate != self.SAMPLE_RATE:
            samples = self._resample(samples, src_rate, self.SAMPLE_RATE)
        return self.transcribe(samples).text

    @property
    def execution_provider(self) -> str:
        return self._ep

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _run_encoder(self, pcm: np.ndarray) -> np.ndarray:
        audio = pcm[np.newaxis, :]  # [1, n_samples]
        outputs = self._encoder.run(
            ["last_hidden_state"],
            {"input_values": audio},
        )
        return outputs[0]  # [1, enc_seq, 288]

    def _decode(self, enc_hidden: np.ndarray):
        """Greedy autoregressive decode with KV-cache.
        Returns (token_ids, time_to_first_token_s).

        Model schema (onnx-community/moonshine-tiny-ONNX):
          n_layers = 6, n_heads = 8, head_dim = 36, vocab = 32768
          Each layer has: past_key_values.L.decoder.{key,value}
                          past_key_values.L.encoder.{key,value}
          The encoder cross-attention KV is filled from present.* on step 0
          and reused unchanged on subsequent steps.
        """
        N_LAYERS   = self._n_kv_layers   # 6 for tiny
        N_HEADS    = 8
        HEAD_DIM   = 36
        ENC_SEQ    = enc_hidden.shape[1]

        token_ids  = []
        cur_token  = np.array([[self.BOS_TOKEN]], dtype=np.int64)
        first_step = True
        ttft       = 0.0

        # KV cache: keyed by exact input name
        # Decoder self-attn: shape [1, 8, past_len, 36]  — grows each step
        # Encoder cross-attn: shape [1, 8, enc_seq, 36]  — fixed after step 0
        kv_cache: dict[str, np.ndarray] = {}

        for step in range(self.MAX_TOKENS):
            feed: dict[str, np.ndarray] = {
                "input_ids":            cur_token,
                "encoder_hidden_states": enc_hidden,
                "use_cache_branch":     np.array([not first_step], dtype=np.bool_),
            }

            for l in range(N_LAYERS):
                # Decoder self-attention past
                dec_k = f"past_key_values.{l}.decoder.key"
                dec_v = f"past_key_values.{l}.decoder.value"
                if dec_k in kv_cache:
                    feed[dec_k] = kv_cache[dec_k]
                    feed[dec_v] = kv_cache[dec_v]
                else:
                    feed[dec_k] = np.zeros((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
                    feed[dec_v] = np.zeros((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)

                # Encoder cross-attention past
                enc_k = f"past_key_values.{l}.encoder.key"
                enc_v = f"past_key_values.{l}.encoder.value"
                if enc_k in kv_cache:
                    feed[enc_k] = kv_cache[enc_k]
                    feed[enc_v] = kv_cache[enc_v]
                else:
                    feed[enc_k] = np.zeros((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
                    feed[enc_v] = np.zeros((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)

            t_step  = time.perf_counter()
            outputs = self._decoder.run(self._dec_output_names, feed)
            if first_step:
                ttft       = time.perf_counter() - t_step
                first_step = False

            # logits: [1, seq_len, vocab] — take last position
            logits     = outputs[0][0, -1]
            next_token = int(np.argmax(logits))
            token_ids.append(next_token)
            cur_token  = np.array([[next_token]], dtype=np.int64)

            if next_token == self.EOS_TOKEN:
                break

            # Update KV cache from present.* outputs
            out_names = self._dec_output_names
            for i, oname in enumerate(out_names):
                kv_name = oname.replace("present.", "past_key_values.")
                kv_cache[kv_name] = outputs[i]

        return token_ids, ttft


    def _decode_tokens(self, ids: list[int]) -> str:
        pieces = []
        for i in ids:
            if i in (self.BOS_TOKEN, self.EOS_TOKEN):
                continue
            piece = self._vocab.get(i, "")
            # GPT-2 BPE: Ġ (U+0120) → space
            piece = piece.replace("\u0120", " ")
            pieces.append(piece)
        return "".join(pieces).strip()

    @staticmethod
    def load_audio(path: Union[str, Path]) -> np.ndarray:
        """Load audio file to 16 kHz mono float32 numpy array."""
        try:
            import soundfile as sf  # type: ignore
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except ImportError:
            raise ImportError(
                "soundfile is required for audio loading. "
                "Install with: pip install soundfile"
            )

        # Mix to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16 kHz
        if sr != MoonshineRunner.SAMPLE_RATE:
            data = MoonshineRunner._resample(data, sr, MoonshineRunner.SAMPLE_RATE)

        return data.astype(np.float32)

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        try:
            import librosa  # type: ignore
            return librosa.resample(audio, orig_sr=src_rate, target_sr=dst_rate)
        except ImportError:
            pass
        # Fallback: linear interpolation
        n_out = int(len(audio) * dst_rate / src_rate)
        indices = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="MoonshineRunner Python binding")
    parser.add_argument("audio", nargs="*", help="Audio file(s) to transcribe")
    parser.add_argument("--model-dir", default="models/moonshine-tiny")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rtf", action="store_true")
    args = parser.parse_args()

    runner = MoonshineRunner(args.model_dir, intra_op_threads=args.threads)

    if not args.audio:
        print("No audio files provided. Specify WAV files as arguments.")
        sys.exit(0)

    for path in args.audio:
        pcm = MoonshineRunner.load_audio(path)
        res = runner.transcribe(pcm)
        print(res.text)
        if args.rtf:
            print(f"  RTF={res.rtf:.3f}  TTFT={res.ttft_ms:.1f}ms", file=sys.stderr)
