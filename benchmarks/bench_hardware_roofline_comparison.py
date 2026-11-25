"""
Hardware Roofline Comparison:
Zen 4 AVX-512 (AMD Ryzen 5 7600X) vs. Ada Lovelace Triton 3.9 (NVIDIA RTX 5060 Ti).
"""

import time
import json
import torch
from weft.kernels.triton_subbit_gemv import triton_subbit_gemv


def run_hardware_roofline_comparison():
    print("=== Hardware Roofline Comparison: Zen 4 CPU vs. Ada Lovelace GPU ===")
    
    # Dimensions corresponding to LLaMA-70B intermediate projections
    B = 1
    K = 3584
    N = 18944
    rank = 64
    words = (rank + 31) // 32
    iterations = 50

    # 1. GPU Benchmark (Triton / CUDA)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x_gpu = torch.randn(B, K, device=device, dtype=torch.float16)
    u_gpu = torch.randint(0, 1000000, (N, words), device=device, dtype=torch.int32)
    v_gpu = torch.randint(0, 1000000, (K, words), device=device, dtype=torch.int32)
    sh_gpu = torch.ones(N, device=device, dtype=torch.float16)
    sl_gpu = torch.ones(rank, device=device, dtype=torch.float16)
    sg_gpu = torch.ones(K, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = triton_subbit_gemv(x_gpu, u_gpu, v_gpu, sh_gpu, sl_gpu, sg_gpu, rank)
    torch.cuda.synchronize()

    start_gpu = time.time()
    for _ in range(iterations):
        _ = triton_subbit_gemv(x_gpu, u_gpu, v_gpu, sh_gpu, sl_gpu, sg_gpu, rank)
    torch.cuda.synchronize()
    elapsed_gpu = time.time() - start_gpu

    total_ops = 2.0 * B * (K * rank + N * rank) * iterations
    gpu_gops = (total_ops / (elapsed_gpu * 1e9))
    gpu_latency_us = (elapsed_gpu / iterations) * 1e6

    print(f"GPU (RTX 5060 Ti / Triton): {gpu_gops:.2f} GOPS, Latency: {gpu_latency_us:.2f} us")

    # 2. CPU Benchmark (PyTorch / AVX-512)
    x_cpu = torch.randn(B, K, dtype=torch.float32)
    u_cpu = torch.randint(0, 1000000, (N, words), dtype=torch.int32)
    v_cpu = torch.randint(0, 1000000, (K, words), dtype=torch.int32)
    sh_cpu = torch.ones(N, dtype=torch.float32)
    sl_cpu = torch.ones(rank, dtype=torch.float32)
    sg_cpu = torch.ones(K, dtype=torch.float32)

    for _ in range(5):
        _ = triton_subbit_gemv(x_cpu, u_cpu, v_cpu, sh_cpu, sl_cpu, sg_cpu, rank)

    start_cpu = time.time()
    for _ in range(iterations):
        _ = triton_subbit_gemv(x_cpu, u_cpu, v_cpu, sh_cpu, sl_cpu, sg_cpu, rank)
    elapsed_cpu = time.time() - start_cpu

    cpu_gops = (total_ops / (elapsed_cpu * 1e9))
    cpu_latency_us = (elapsed_cpu / iterations) * 1e6

    print(f"CPU (Ryzen 5 7600X / Zen 4): {cpu_gops:.2f} GOPS, Latency: {cpu_latency_us:.2f} us")

    results = {
        "workload": {"B": B, "K": K, "N": N, "rank": rank, "iterations": iterations},
        "cpu_ryzen_7600x": {
            "giga_ops_per_second": cpu_gops,
            "latency_microseconds": cpu_latency_us,
            "elapsed_seconds": elapsed_cpu
        },
        "gpu_rtx_5060_ti": {
            "giga_ops_per_second": gpu_gops,
            "latency_microseconds": gpu_latency_us,
            "elapsed_seconds": elapsed_gpu,
            "speedup_vs_cpu": cpu_latency_us / max(gpu_latency_us, 1e-6)
        }
    }

    with open("results/hardware_roofline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[PASS] Results written to results/hardware_roofline_comparison.json")


if __name__ == "__main__":
    run_hardware_roofline_comparison()
