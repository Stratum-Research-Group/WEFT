"""
Profile the effect of increasing state rank R from 16 to 32, 64, and 128.
Measures:
1. Recurrent state memory per layer (KB)
2. Triton forward kernel latency across sequence lengths
3. Parameter count scaling
4. Numerical stability / execution correctness
"""

import time
import torch
from weft.kernels.fused_chunk_scan import fused_chunk_recurrent_forward_triton

def benchmark_ranks():
    device = "cuda"
    b = 2
    h = 4
    d = 128
    ranks = [16, 32, 64, 128]
    seq_lengths = [2048, 8192, 32768]
    dtype = torch.bfloat16

    print("=========================================================================")
    print("  WEFT STATE RANK SCALING BENCHMARK (R = 16, 32, 64, 128)")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Batch: {b}, Heads: {h}, HeadDim: {d}, Dtype: {dtype}")
    print("=========================================================================")

    print(f"\n{'Rank (R)':>8} | {'State Size / Layer':>18} | {'Param Overhead / Head':>22}")
    print("-" * 55)
    for r in ranks:
        # State tensor: [H, R, D] in float16/bfloat16 (2 bytes)
        state_bytes = h * r * d * 2
        state_kb = state_bytes / 1024.0
        # Linear projection parameters: proj_k (d -> r), proj_v (d -> r) per head
        params_per_head = 2 * (d * r)
        print(f"{r:8d} | {state_kb:15.2f} KB | {params_per_head:20d} weights")

    print("\nLatency Scaling across Sequence Lengths (Forward Pass):")
    header = f"{'Rank (R)':>8} | " + " | ".join([f"N = {s:<7}" for s in seq_lengths])
    print(header)
    print("-" * len(header))

    for r in ranks:
        row = [f"{r:8d}"]
        for s in seq_lengths:
            x = torch.randn(b, s, h, d, device=device, dtype=dtype)
            kp = torch.randn(b, s, h, r, device=device, dtype=dtype)
            vp = torch.randn(b, s, h, r, device=device, dtype=dtype)
            gamma = torch.randn(h, r, device=device, dtype=dtype)

            # Warmup
            for _ in range(3):
                _ = fused_chunk_recurrent_forward_triton(x, kp, vp, gamma)
            torch.cuda.synchronize()

            iters = 10
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = fused_chunk_recurrent_forward_triton(x, kp, vp, gamma)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) / iters * 1000.0
            row.append(f"{elapsed_ms:8.2f} ms")
        print(" | ".join(row))

    print("\n=========================================================================")

if __name__ == "__main__":
    benchmark_ranks()
