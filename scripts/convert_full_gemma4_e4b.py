"""
Complete Gemma-4 E4B-it Full-Model Sub-Bit Quantization Script:
Quantizes all linear projections of Gemma-4 E4B-it (15 GB raw weights)
to an external certified .weft and .gguf container with zero disk footprint on /home.
"""

import os
import sys
import time
import json
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft.converter import convert_model_to_weft
from weft.export_subbit_gguf import export_weft_to_gguf as convert_weft_to_gguf

MODEL_PATH = "/run/media/akarta/AI/models/gemma-4-E4B-it/model.safetensors"
CONFIG_PATH = "/run/media/akarta/AI/models/gemma-4-E4B-it/config.json"
OUTPUT_DIR = "/run/media/akarta/AI/models"
OUTPUT_WEFT = os.path.join(OUTPUT_DIR, "gemma-4-E4B-it-subbit.weft")
OUTPUT_GGUF = os.path.join(OUTPUT_DIR, "gemma-4-E4B-it-subbit.gguf")


def run_conversion():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    print("=" * 80)
    print("  GEMMA-4 E4B-IT COMPLETE SUB-BIT QUANTIZATION & CERTIFICATION")
    print("=" * 80)
    print(f"Source: {MODEL_PATH}")
    print(f"Target WEFT: {OUTPUT_WEFT}")
    print(f"Target GGUF: {OUTPUT_GGUF}")

    t0 = time.time()
    state_dict = {}
    total_dense_bytes = 0

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight") and ("mlp" in k or "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k or "linear" in k):
                t = f.get_tensor(k)
                if t.ndim == 2:
                    state_dict[k] = t
                    total_dense_bytes += t.numel() * t.element_size()

    print(f"Discovered {len(state_dict)} 2D linear weight matrices ({total_dense_bytes / (1024**3):.2f} GB dense FP16).")
    print(f"Executing LittleBit-2 Joint-ITQ factorization & interval certification...")

    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as cf:
            config = json.load(cf)

    config["architecture"] = "gemma-4-e4b-it"
    config["num_quantized_layers"] = len(state_dict)

    container, certificate = convert_model_to_weft(
        state_dict=state_dict,
        config=config,
        output_filepath=OUTPUT_WEFT,
        base_rank=64,
        adaptive_rank=True,
        certify_intervals=True,
    )

    cert_path = os.path.join(OUTPUT_DIR, "gemma-4-E4B-it_certificate.json")
    if certificate:
        with open(cert_path, "w") as cf:
            json.dump(certificate, cf, indent=2)
        print(f"Lyapunov interval certificate saved to {cert_path}")

    # Generate matching consolidated GGUF
    print(f"Exporting matching GGUF container -> {OUTPUT_GGUF}...")
    convert_weft_to_gguf(OUTPUT_WEFT, OUTPUT_GGUF)

    elapsed = time.time() - t0
    weft_size = os.path.getsize(OUTPUT_WEFT) / (1024**2)
    gguf_size = os.path.getsize(OUTPUT_GGUF) / (1024**2) if os.path.exists(OUTPUT_GGUF) else 0

    print("=" * 80)
    print(f"Conversion completed successfully in {elapsed:.1f}s.")
    print(f"Output .weft size: {weft_size:.2f} MB")
    print(f"Output .gguf size: {gguf_size:.2f} MB")
    print(f"Compression ratio: {(total_dense_bytes / (weft_size * 1024 * 1024)):.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    run_conversion()
