#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
# SPDX-License-Identifier: Apache-2.0
#
# scripts/export_onnx.py
#
# Export Moonshine from HuggingFace to ONNX and apply INT8 quantization.
#
# Usage:
#   pip install moonshine-onnx optimum[exporters] onnxruntime
#   python scripts/export_onnx.py --model tiny --output models/moonshine-tiny

"""
Export Moonshine models to quantized ONNX format.

The exported files are:
  encoder_model_int8.onnx
  decoder_model_merged_int8.onnx
  tokenizer.json
  tokenizer_config.json
"""

import argparse
import os
import shutil
from pathlib import Path


def export_moonshine(model_size: str, output_dir: Path) -> None:
    """Export Moonshine to ONNX using the optimum CLI, then quantize."""
    model_id = f"onnx-community/moonshine-{model_size}-ONNX"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Downloading pre-exported ONNX from: {model_id}")
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed. Run: pip install huggingface-hub"
        )

    # Files to grab (stored in onnx/ subdir in onnx-community repos)
    needed = {
        "onnx/encoder_model_int8.onnx":         "encoder_model_int8.onnx",
        "onnx/decoder_model_merged_int8.onnx":   "decoder_model_merged_int8.onnx",
        "tokenizer.json":                        "tokenizer.json",
        "tokenizer_config.json":                 "tokenizer_config.json",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for remote_path, local_name in needed.items():
        dest = output_dir / local_name
        if dest.exists():
            print(f"  Cached: {local_name}")
            continue
        print(f"  Downloading {local_name} ...")
        downloaded = hf_hub_download(
            repo_id=model_id,
            filename=remote_path,
            local_dir=str(output_dir),
        )
        # hf_hub_download places file at output_dir/onnx/<name>; move it up
        import shutil
        dl_path = Path(downloaded)
        if dl_path.name != local_name:
            shutil.move(str(dl_path), str(dest))
        print(f"  OK: {local_name}")

    print("[2/3] Applying INT8 dynamic quantization...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore
    except ImportError:
        raise SystemExit(
            "onnxruntime not installed. Run: pip install onnxruntime"
        )

    onnx_files = list(output_dir.glob("*.onnx"))
    for onnx_path in onnx_files:
        if onnx_path.stem.endswith("_int8"):
            continue  # already quantized
        out_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        if out_path.exists():
            print(f"      Skip (cached): {out_path.name}")
            continue
        print(f"      Quantizing {onnx_path.name} → {out_path.name}")
        quantize_dynamic(
            str(onnx_path),
            str(out_path),
            weight_type=QuantType.QInt8,
        )
        # Keep original for reference, remove to save disk space:
        # onnx_path.unlink()

    print("[3/3] Verifying outputs...")
    required = [
        "encoder_model_int8.onnx",
        "decoder_model_merged_int8.onnx",
        "tokenizer.json",
    ]
    missing = [r for r in required if not (output_dir / r).exists()]
    if missing:
        # Try alternative naming (some model versions differ)
        alt_missing = []
        for m in missing:
            stem = m.replace("_int8.onnx", ".onnx")
            if not (output_dir / stem).exists():
                alt_missing.append(m)
        if alt_missing:
            print(f"WARNING: Missing files: {alt_missing}")
            print("         The model dir may use different filenames.")
            print("         Check the HuggingFace repo for exact file names.")
    else:
        print("All required files present.")

    print(f"\nDone! Model directory: {output_dir.resolve()}")
    _print_sizes(output_dir)


def _print_sizes(directory: Path) -> None:
    """Print file sizes for the output directory."""
    print("\nFile sizes:")
    total = 0
    for f in sorted(directory.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total += size_mb
        print(f"  {f.name:<50} {size_mb:6.1f} MB")
    print(f"  {'Total ONNX':<50} {total:6.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Moonshine ONNX models with INT8 quantization"
    )
    parser.add_argument(
        "--model", choices=["tiny", "base"], default="tiny",
        help="Model variant (default: tiny)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: models/moonshine-<size>)"
    )
    args = parser.parse_args()

    out = args.output or Path("models") / f"moonshine-{args.model}"
    export_moonshine(args.model, out)


if __name__ == "__main__":
    main()
