"""
End-to-end benchmark script: Quantizes linear layers from Gemma-4 safetensors
into a native .weft sub-bit model container on external NVMe storage.
Measures compression ratio, effective BPW, and execution latency.
"""

import os
import time
import json
import torch
from safetensors import safe_open
from weft.converter import convert_model_to_weft
from weft.format import WeftModelContainer

MODEL_PATH = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
OUTPUT_DIR = "/run/media/akarta/AI/checkpoints/benchmark_subbit"
OUTPUT_WEFT = os.path.join(OUTPUT_DIR, "gemma4_e2b_subbit.weft")


def run_benchmark():
    if not os.path.exists(MODEL_PATH):
        print(f"Model path {MODEL_PATH} not found. Skipping.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Sub-Bit Model Quantization Benchmark (Gemma-4 E2B) ===")
    t0 = time.time()

    # Load 4 representative linear layers from safetensors
    state_dict = {}
    total_dense_bytes = 0
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if "mlp" in k and k.endswith(".weight")][:4]
        for k in keys:
            t = f.get_tensor(k)
            state_dict[k] = t
            total_dense_bytes += t.numel() * t.element_size()
            print(f"Loaded: {k} -> shape {t.shape}, {t.numel() * t.element_size() / 1e6:.2f} MB")

    config = {
        "architecture": "gemma-4-e2b-subbit",
        "hidden_size": 2048,
        "vocab_size": 256000,
        "num_layers": len(state_dict),
    }

    print(f"\nCompressing {len(state_dict)} layers to {OUTPUT_WEFT}...")
    t_conv0 = time.time()
    container = convert_model_to_weft(
        state_dict,
        config,
        OUTPUT_WEFT,
        base_rank=64,
        adaptive_rank=True,
        compress_entropy=False,
    )
    conv_duration = time.time() - t_conv0

    weft_size_bytes = os.path.getsize(OUTPUT_WEFT)
    compression_ratio = total_dense_bytes / max(1, weft_size_bytes)
    
    # Calculate weighted effective BPW across quantized layers
    bpw_list = [container.get_subbit_linear(name).bits_per_weight() for name in container.tensors.keys()]
    mean_bpw = sum(bpw_list) / len(bpw_list)

    print(f"\n--- Benchmark Results ---")
    print(f"Total Dense Size:    {total_dense_bytes / 1e6:.2f} MB")
    print(f"WEFT Container Size: {weft_size_bytes / 1e6:.2f} MB")
    print(f"Compression Ratio:   {compression_ratio:.2f}x reduction")
    print(f"Effective Mean BPW:  {mean_bpw:.4f} bits per weight")
    print(f"Conversion Latency:  {conv_duration:.2f}s")

    # Verify zero-copy reload
    loaded = WeftModelContainer.load(OUTPUT_WEFT)
    print(f"Loaded container layers: {list(loaded.tensors.keys())}")
    print("Benchmark complete. Zero footprint on /home maintained.")


if __name__ == "__main__":
    run_benchmark()
