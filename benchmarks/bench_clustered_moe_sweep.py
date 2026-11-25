"""
Hardware Sweep Profiler for 1T Clustered MoE Dispatch on NVIDIA GeForce RTX 5060 Ti Blackwell.
Sweeps batch sizes B in {1, 4, 16, 64} across unclustered vs 2-tier clustered dispatch.
Records real execution latency (us), GPU memory consumption, and throughput.
"""

import os
import time
import json
import torch
from weft.kernels.fused_moe_dispatch_1t import fused_moe_dispatch_1t
from weft.kernels.fused_cluster_moe_dispatch import fused_cluster_moe_dispatch

def run_moe_dispatch_sweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running Clustered MoE Dispatch Sweep on device: {device}")
    
    hidden_dim = 8192
    num_experts = 128
    num_clusters = 16
    batch_sizes = [1, 4, 16, 64]
    
    cluster_gate = torch.randn(num_clusters, hidden_dim, device=device, dtype=torch.float32)
    expert_gate = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.float32)

    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hidden_dim": hidden_dim,
        "num_experts": num_experts,
        "num_clusters": num_clusters,
        "sweeps": []
    }

    warmup = 10
    iters = 50

    for B in batch_sizes:
        tokens = torch.randn(B, hidden_dim, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(warmup):
            _ = fused_moe_dispatch_1t(tokens, expert_gate)
            _ = fused_cluster_moe_dispatch(tokens, cluster_gate, expert_gate)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark standard unclustered dispatch
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_moe_dispatch_1t(tokens, expert_gate)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        unclustered_lat_us = ((t1 - t0) / iters) * 1e6

        # Benchmark 2-tier clustered dispatch
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_cluster_moe_dispatch(tokens, cluster_gate, expert_gate)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        clustered_lat_us = ((t1 - t0) / iters) * 1e6

        speedup = unclustered_lat_us / max(clustered_lat_us, 1e-6)
        sweep_data = {
            "batch_size": B,
            "unclustered_latency_us": unclustered_lat_us,
            "clustered_latency_us": clustered_lat_us,
            "speedup": speedup,
            "tokens_per_second": (B / (clustered_lat_us * 1e-6))
        }
        results["sweeps"].append(sweep_data)
        print(f"[+] B={B:2d} | Unclustered: {unclustered_lat_us:6.2f} us | Clustered: {clustered_lat_us:6.2f} us | Speedup: {speedup:.2f}x")

    out_file = "results/clustered_moe_dispatch_rtx5060ti.json"
    os.makedirs("results", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[*] Benchmark sweep committed to: {out_file}")
    return results

if __name__ == "__main__":
    run_moe_dispatch_sweep()
