"""
Gemma-4 26B Extreme-Scale Sub-Bit Quantization Benchmark:
Streams through sharded safetensors of Gemma-4 26B on external NVMe
and quantizes linear projection matrices into a compact .weft container.
"""

import os
import time
import json
import torch
from safetensors import safe_open
from weft.converter import convert_model_to_weft
from weft.format import WeftModelContainer

SHARD_PATH = "/run/media/akarta/AI/models/gemma-4-26B-A4B-it/model-00002-of-00002.safetensors"
OUTPUT_DIR = "/run/media/akarta/AI/checkpoints/gemma4_26b_subbit"
OUTPUT_WEFT = os.path.join(OUTPUT_DIR, "gemma4_26b_shard2_subbit.weft")


def run_26b_quantization():
    if not os.path.exists(SHARD_PATH):
        print(f"File {SHARD_PATH} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("==================================================================")
    print("  GEMMA-4 26B EXTREME-SCALE SUB-BIT QUANTIZATION BENCHMARK")
    print("==================================================================")
    print(f"Reading from: {SHARD_PATH}")
    print(f"Output to:   {OUTPUT_WEFT}")

    t0 = time.time()
    state_dict = {}
    total_dense_bytes = 0

    with safe_open(SHARD_PATH, framework="pt", device="cpu") as f:
        # Load sample 2D weight matrices from the 26B shard
        keys = [k for k in f.keys() if k.endswith(".weight")][:6]
        for k in keys:
            t = f.get_tensor(k)
            if t.ndim == 2:
                state_dict[k] = t
                total_dense_bytes += t.numel() * t.element_size()
                print(f"Loaded: {k} -> shape {t.shape}, {t.numel() * t.element_size() / 1e6:.2f} MB")

    config = {
        "architecture": "gemma-4-26b-a4b-it",
        "num_quantized_layers": len(state_dict),
    }

    print(f"\nCompressing {len(state_dict)} layers from 26B architecture...")
    container = convert_model_to_weft(
        state_dict=state_dict,
        config=config,
        output_filepath=OUTPUT_WEFT,
        base_rank=64,
        adaptive_rank=True,
    )
    duration = time.time() - t0

    weft_size_bytes = os.path.getsize(OUTPUT_WEFT)
    ratio = total_dense_bytes / max(1, weft_size_bytes)
    bpw_list = [container.get_subbit_linear(k[:-len(".weight")]).bits_per_weight() for k in state_dict.keys()]
    mean_bpw = sum(bpw_list) / len(bpw_list)

    print("\n--- 26B Quantization Results ---")
    print(f"Dense FP16 Size:    {total_dense_bytes / 1e6:.2f} MB")
    print(f"WEFT Sub-Bit Size:  {weft_size_bytes / 1e6:.2f} MB")
    print(f"Compression Ratio:  {ratio:.2f}x reduction")
    print(f"Mean Effective BPW: {mean_bpw:.4f} bits/weight")
    print(f"Runtime:            {duration:.2f}s")
    print("Zero storage footprint on /home.")


if __name__ == "__main__":
    run_26b_quantization()
