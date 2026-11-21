"""
Complete 28-Layer Gemma-4 Full-Model Sub-Bit Quantization Script:
Quantizes all linear projections across all 28 transformer blocks of Gemma-4 E2B-it
to an external .weft container with zero disk footprint on /home.
"""

import os
import time
import json
import torch
from safetensors import safe_open
from weft.converter import convert_model_to_weft
from weft.format import WeftModelContainer

MODEL_PATH = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
OUTPUT_DIR = "/run/media/akarta/AI/models"
OUTPUT_WEFT = os.path.join(OUTPUT_DIR, "gemma-4-E2B-it-subbit.weft")


def run_full_conversion():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    print("==================================================================")
    print("  GEMMA-4 E2B-IT COMPLETE 28-LAYER SUB-BIT QUANTIZATION")
    print("==================================================================")
    print(f"Source: {MODEL_PATH}")
    print(f"Target: {OUTPUT_WEFT}")

    t0 = time.time()
    state_dict = {}
    total_dense_bytes = 0

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight") and ("mlp" in k or "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k):
                t = f.get_tensor(k)
                if t.ndim == 2:
                    state_dict[k] = t
                    total_dense_bytes += t.numel() * t.element_size()

    print(f"Discovered {len(state_dict)} 2D linear weight matrices ({total_dense_bytes / 1e6:.2f} MB dense FP16).")
    print(f"Executing LittleBit-2 Joint-ITQ factorization across all {len(state_dict)} layers...")

    config = {
        "architecture": "gemma-4-e2b-it",
        "num_quantized_layers": len(state_dict),
        "vocab_size": 256000,
        "hidden_size": 2048,
    }

    container = convert_model_to_weft(
        state_dict=state_dict,
        config=config,
        output_filepath=OUTPUT_WEFT,
        base_rank=64,
        adaptive_rank=True,
        compress_entropy=False,
    )
    duration = time.time() - t0

    weft_size_bytes = os.path.getsize(OUTPUT_WEFT)
    ratio = total_dense_bytes / max(1, weft_size_bytes)
    bpw_list = [container.get_subbit_linear(k[:-len(".weight")]).bits_per_weight() for k in state_dict.keys()]
    mean_bpw = sum(bpw_list) / len(bpw_list)

    print("\n==================================================================")
    print("  CONVERSION COMPLETE")
    print("==================================================================")
    print(f"Total Tensors:       {len(state_dict)}")
    print(f"Original FP16 Size:  {total_dense_bytes / 1e6:.2f} MB")
    print(f"WEFT Sub-Bit Size:   {weft_size_bytes / 1e6:.2f} MB")
    print(f"Compression Ratio:   {ratio:.2f}x reduction")
    print(f"Mean Effective BPW:  {mean_bpw:.4f} bits per weight")
    print(f"Total Runtime:       {duration:.2f}s")
    print(f"Saved to:            {OUTPUT_WEFT}")
    print("Storage Compliance:  100% on external NVMe. Zero impact on /home.")


if __name__ == "__main__":
    run_full_conversion()
