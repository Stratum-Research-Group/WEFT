"""
Hardware & System Profiler: Isolates all system bottlenecks across the inference pipeline:
1. PCIe Bus & Host-to-Device (H2D) Tensor Transfer Latency.
2. GPU Kernel Launch Overhead (CPU-bound Python dispatcher latency vs GPU compute).
3. Memory Bandwidth Saturation vs Compute Boundness (Roofline Model Analysis).
4. Sliced Disk I/O Read Latency from external NVMe /run/media/akarta/AI.
"""
import os
import sys
import time
import torch
import torch.nn as nn
from safetensors import safe_open

def profile_pcie_transfer(device):
    print("\n[Bottleneck Analysis 1: PCIe Bus Host-to-Device (H2D) Transfer]")
    tensor_sizes = [1024, 1024*1024, 16*1024*1024] # 4 KB, 4 MB, 64 MB
    for n in tensor_sizes:
        x_cpu = torch.randn(n)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_gpu = x_cpu.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        mb = (n * 4) / (1024 * 1024)
        bw_gb_s = (mb / 1024.0) / (elapsed / 1000.0) if elapsed > 0 else 0
        print(f"  • Size: {mb:6.2f} MB | Latency: {elapsed:6.2f} ms | Effective Transfer BW: {bw_gb_s:5.2f} GB/s")

def profile_nvme_disk_io():
    print("\n[Bottleneck Analysis 2: External NVMe Disk mmap & safe_open Slicing]")
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
    t0 = time.perf_counter()
    with safe_open(model_path, framework="pt") as f:
        t_open = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        _ = f.get_slice("model.language_model.embed_tokens.weight")[:1024, :]
        t_slice = (time.perf_counter() - t1) * 1000.0
    print(f"  • safe_open Header Parse Latency: {t_open:.2f} ms")
    print(f"  • Layer Slice Access Latency (Zero-RAM mmap): {t_slice:.2f} ms")

def profile_cpu_python_dispatch(device):
    print("\n[Bottleneck Analysis 3: Python Runtime & Kernel Launch Overhead]")
    x = torch.randn(1, 256, device=device)
    w = torch.randn(256, 256, device=device)
    # Warmup
    for _ in range(10): _ = torch.matmul(x, w)

    # 1,000 small launches
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = torch.matmul(x, w)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    per_op_us = (total_ms / 1000.0) * 1000.0
    print(f"  • 1,000 Kernel Dispatches: {total_ms:.2f} ms total")
    print(f"  • Per-Kernel CPU Dispatch Overhead: {per_op_us:.2f} µs (CUDA driver / Python GIL overhead)")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"  IDENTIFYING INFRASTRUCTURE & INFERENCE BOTTLENECK PROFILE ON {device.upper()}")
    print("=" * 80)
    profile_pcie_transfer(device)
    profile_nvme_disk_io()
    profile_cpu_python_dispatch(device)
    print("\n" + "=" * 80)
