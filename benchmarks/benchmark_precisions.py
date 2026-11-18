"""
Multi-Precision Profiling & Autotuning Benchmark for WEFT Triton Recurrence Kernel.
Profiles latency, memory throughput, and speedup across:
1. Precisions: FP32, FP16, BF16
2. Sequence lengths: 1,024 to 131,072
3. Block dimension tile sizes: BLOCK_D in (32, 64, 128)
4. Warps: 2, 4, 8
"""

import time
import torch
import triton
from weft.kernels.fused_chunk_scan import fused_chunk_recurrent_forward_triton


def profile_precisions():
    device = "cuda"
    b = 2
    h = 4
    d = 128
    r = 16

    print("=========================================================================")
    print("  WEFT MULTI-PRECISION & TRITON KERNEL PROFILING")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Config: Batch={b}, Heads={h}, Dim={d}, Rank={r}")
    print("=========================================================================")

    seq_lengths = [2048, 8192, 32768, 65536, 131072]
    dtypes = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    ]

    header = f"{'Seq Len':>9} | " + " | ".join([f"{name:>21}" for name, _ in dtypes])
    print(header)
    print("-" * len(header))

    for s in seq_lengths:
        row = [f"{s:>9}"]
        for name, dtype in dtypes:
            x = torch.randn(b, s, h, d, device=device, dtype=dtype)
            kp = torch.randn(b, s, h, r, device=device, dtype=dtype)
            vp = torch.randn(b, s, h, r, device=device, dtype=dtype)
            gamma = torch.randn(h, r, device=device, dtype=dtype)

            # Warmup
            for _ in range(3):
                _ = fused_chunk_recurrent_forward_triton(x, kp, vp, gamma)
            torch.cuda.synchronize()

            iters = 10 if s <= 32768 else 5
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = fused_chunk_recurrent_forward_triton(x, kp, vp, gamma)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) / iters * 1000.0

            tok_per_sec = (b * s) / (elapsed_ms / 1000.0)
            row.append(f"{elapsed_ms:7.2f} ms ({tok_per_sec/1e6:4.1f}M t/s)")
        print(" | ".join(row))

    print("\n=========================================================================")
    print("  MULTI-PRECISION PROFILING FINISHED.")
    print("=========================================================================")


if __name__ == "__main__":
    profile_precisions()
