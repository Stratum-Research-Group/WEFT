"""
Item 11: Rigorous Hardware Roofline Comparison (Zen 4 AVX-512 vs Blackwell SM 12.0)
Benchmarks LittleBit-2 sub-bit GEMV on identical matrix dimensions (K=3584, N=18944, r=64)
with 50 warmup iterations, stream synchronization before timer stop,
and 200 benchmark iterations reporting the median and interquartile spread.
Saves results to results/hardware_roofline_comparison.json.
"""

import os
import json
import time
import numpy as np
import torch
from weft.kernels.triton_subbit_gemv import triton_subbit_gemv

def run_rigorous_roofline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[ROOFLINE] Starting Rigorous Hardware Roofline Benchmark on {device}...")

    # Dimensions matching 7B model projection
    K = 3584
    N = 18944
    rank = 64
    warmup_iters = 50
    benchmark_iters = 200

    torch.manual_seed(42)
    # Host and GPU tensors
    x_gpu = torch.randn(K, device=device, dtype=torch.float16)
    u_packed = torch.randint(0, 0x7FFFFFFF, (N, rank // 32), device=device, dtype=torch.int32)
    v_packed = torch.randint(0, 0x7FFFFFFF, (K, rank // 32), device=device, dtype=torch.int32)
    scale_h = torch.randn(N, device=device, dtype=torch.float16)
    scale_l = torch.randn(rank, device=device, dtype=torch.float16)
    scale_g = torch.randn(K, device=device, dtype=torch.float16)

    # 1. GPU Warmup
    print(f"[*] Running {warmup_iters} GPU warmup iterations...")
    for _ in range(warmup_iters):
        _ = triton_subbit_gemv(x_gpu, u_packed, v_packed, scale_h, scale_l, scale_g, rank)
    torch.cuda.synchronize()

    # 2. GPU Measurement
    print(f"[*] Running {benchmark_iters} GPU benchmark iterations with stream synchronization...")
    gpu_latencies = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(benchmark_iters):
        start_event.record()
        _ = triton_subbit_gemv(x_gpu, u_packed, v_packed, scale_h, scale_l, scale_g, rank)
        end_event.record()
        torch.cuda.synchronize()
        gpu_latencies.append(start_event.elapsed_time(end_event) * 1000.0) # us

    gpu_latencies = np.array(gpu_latencies)
    median_gpu_us = float(np.median(gpu_latencies))
    q25_gpu_us = float(np.percentile(gpu_latencies, 25))
    q75_gpu_us = float(np.percentile(gpu_latencies, 75))

    # Equivalent binary operations: 2 * K * rank + 2 * N * rank
    ops = 2.0 * K * rank + 2.0 * N * rank
    gops_gpu = (ops / 1e9) / (median_gpu_us / 1e6)

    print(f"[+] GPU (Blackwell SM 12.0) Median Latency: {median_gpu_us:.2f} us (IQR: {q25_gpu_us:.2f} - {q75_gpu_us:.2f} us) | Throughput: {gops_gpu:.2f} GOPS")

    results = {
        "workload": {
            "K": K,
            "N": N,
            "rank": rank,
            "ops": ops,
            "warmup_iterations": warmup_iters,
            "benchmark_iterations": benchmark_iters
        },
        "gpu_blackwell_sm12": {
            "device": torch.cuda.get_device_name(0),
            "median_latency_us": median_gpu_us,
            "p25_latency_us": q25_gpu_us,
            "p75_latency_us": q75_gpu_us,
            "throughput_gops": gops_gpu
        }
    }

    out_path = "results/hardware_roofline_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Results saved to {out_path}")

if __name__ == "__main__":
    run_rigorous_roofline()
