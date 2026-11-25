"""
Real Conversion Benchmark for Qwen2.5-Coder-0.5B Safetensors into WEFT Format.
Measures true disk read throughput, factorisation time, peak RSS memory,
and verifies storage safety invariants without mock or synthetic tensors.
"""

import os
import sys
import time
import json
import psutil
import torch
from safetensors import safe_open
from weft.converter import convert_model_to_weft
from weft.convert_llama_405b_streaming import verify_storage_safety

MODEL_SAFETENSORS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301/model.safetensors"
)
OUTPUT_WEFT = "/run/media/akarta/AI/models/qwen2.5-coder-0.5b-benchmark.weft"

def benchmark_real_conversion():
    ai_free_start, home_free_start = verify_storage_safety()
    process = psutil.Process()
    start_rss = process.memory_info().rss / (1024**2)

    if not os.path.exists(MODEL_SAFETENSORS):
        raise FileNotFoundError(f"Source model weights not found at: {MODEL_SAFETENSORS}")

    source_bytes = os.path.getsize(MODEL_SAFETENSORS)
    source_mb = source_bytes / (1024**2)

    print(f"[*] Loading state_dict from physical safetensors ({source_mb:.2f} MB)...")
    t0 = time.perf_counter()
    state_dict = {}
    with safe_open(MODEL_SAFETENSORS, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    t_load = time.perf_counter() - t0

    num_linear = sum(1 for k, v in state_dict.items() if k.endswith(".weight") and v.ndim == 2)
    print(f"[+] Loaded {len(state_dict)} tensors ({num_linear} 2D linear projections) in {t_load:.2f} s")

    # Run actual WEFT factorisation on real weight matrices
    t1 = time.perf_counter()
    container, cert = convert_model_to_weft(
        state_dict=state_dict,
        config={"architecture": "qwen2.5-coder-0.5b", "source": "safetensors"},
        output_filepath=OUTPUT_WEFT,
        base_rank=32,
        adaptive_rank=True,
        certify_intervals=False
    )
    container.save(OUTPUT_WEFT)
    t_convert = time.perf_counter() - t1
    total_elapsed = t_load + t_convert

    out_bytes = os.path.getsize(OUTPUT_WEFT)
    out_mb = out_bytes / (1024**2)
    peak_rss = process.memory_info().rss / (1024**2)
    ai_free_end, home_free_end = verify_storage_safety()

    results = {
        "model": "Qwen2.5-Coder-0.5B",
        "source_safetensors_mb": source_mb,
        "converted_weft_mb": out_mb,
        "compression_ratio": source_mb / max(out_mb, 1e-6),
        "linear_layers_converted": num_linear,
        "load_time_seconds": t_load,
        "factorize_time_seconds": t_convert,
        "total_elapsed_seconds": total_elapsed,
        "overall_throughput_mb_s": source_mb / total_elapsed,
        "peak_rss_mb": peak_rss,
        "storage_floor_preserved": ai_free_end >= 700.0 and home_free_end >= 198.0
    }

    # Clean up benchmark target artifact to maintain storage hygiene
    if os.path.exists(OUTPUT_WEFT):
        os.remove(OUTPUT_WEFT)

    os.makedirs("results", exist_ok=True)
    out_file = "results/real_conversion_qwen_0.5b.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=================================================================")
    print("  REAL QWEN-0.5B SAFETENSORS CONVERSION BENCHMARK RESULTS")
    print("=================================================================")
    print(f"Source Model Size    : {source_mb:.2f} MB")
    print(f"Converted WEFT Size  : {out_mb:.2f} MB (Compression: {results['compression_ratio']:.2f}x)")
    print(f"Factorization Time   : {t_convert:.2f} s ({num_linear} linear layers)")
    print(f"Throughput           : {results['overall_throughput_mb_s']:.2f} MB/s")
    print(f"Peak RSS Memory      : {peak_rss:.2f} MB")
    print(f"Floor Preserved      : {results['storage_floor_preserved']}")
    print(f"Committed Artifact   : {out_file}")
    print("=================================================================")
    return results

if __name__ == "__main__":
    benchmark_real_conversion()
