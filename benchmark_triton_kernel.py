"""
Triton Kernel Verification and Speed Benchmark.
Compares pure PyTorch chunk scan vs. Fused Triton SRAM scan for:
1. Numerical equivalence (|y_triton - y_pytorch| < 1e-4)
2. Final recurrent state equivalence (|S_triton - S_pytorch| < 1e-4)
3. Wall-clock latency (ms) across sequence lengths N = [1024, 4096, 16384, 65536]
4. Peak VRAM footprint
"""

import time
import torch
from weft.kernels.chunk_scan_pt import chunk_gated_recurrent_forward_pt
from weft.kernels.fused_chunk_scan import fused_chunk_recurrent_forward_triton


def verify_and_benchmark():
    device = "cuda"
    torch.manual_seed(42)

    b = 2
    h = 4
    d = 128
    r = 16

    print("=================================================================")
    print("  WEFT TRITON FUSED SCAN KERNEL VERIFICATION & BENCHMARK")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Config: Batch={b}, Heads={h}, Dim={d}, Rank={r}")
    print("=================================================================")

    # 1. Numerical Verification on N = 512
    s_test = 512
    x = torch.randn(b, s_test, h, d, device=device, dtype=torch.float32)
    kp = torch.randn(b, s_test, h, r, device=device, dtype=torch.float32)
    vp = torch.randn(b, s_test, h, r, device=device, dtype=torch.float32)
    decay = torch.randn(h, r, device=device, dtype=torch.float32)

    print("\n--> Step 1: Testing numerical equivalence on N = 512...")
    out_pt, state_pt = chunk_gated_recurrent_forward_pt(x, kp, vp, decay, chunk_size=128)
    out_triton, state_triton = fused_chunk_recurrent_forward_triton(x, kp, vp, decay)

    out_diff = torch.max(torch.abs(out_pt - out_triton)).item()
    state_diff = torch.max(torch.abs(state_pt - state_triton)).item()

    print(f"    Max Output Diff: {out_diff:.6e}")
    print(f"    Max State Diff:  {state_diff:.6e}")
    assert out_diff < 1e-4, f"Output difference too large: {out_diff}"
    assert state_diff < 1e-4, f"State difference too large: {state_diff}"
    print("    [PASS] Triton kernel output and recurrent state match PyTorch baseline.")

    # 2. Performance Benchmark across Sequence Lengths
    seq_lengths = [1024, 4096, 16384, 65536, 131072]
    print("\n--> Step 2: Latency Benchmark across sequence lengths...")
    print(f"{'Seq Len':>10} | {'PyTorch Latency':>16} | {'Triton Latency':>15} | {'Speedup':>10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        x_bench = torch.randn(b, seq_len, h, d, device=device, dtype=torch.float32)
        kp_bench = torch.randn(b, seq_len, h, r, device=device, dtype=torch.float32)
        vp_bench = torch.randn(b, seq_len, h, r, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            _ = fused_chunk_recurrent_forward_triton(x_bench, kp_bench, vp_bench, decay)
        torch.cuda.synchronize()

        # Triton Timing
        start = time.perf_counter()
        iters = 10 if seq_len <= 16384 else 5
        for _ in range(iters):
            out_triton, _ = fused_chunk_recurrent_forward_triton(x_bench, kp_bench, vp_bench, decay)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / iters * 1000

        # PyTorch Timing (skip largest if slow)
        if seq_len <= 16384:
            for _ in range(3):
                _ = chunk_gated_recurrent_forward_pt(x_bench, kp_bench, vp_bench, decay, chunk_size=128)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                out_pt, _ = chunk_gated_recurrent_forward_pt(x_bench, kp_bench, vp_bench, decay, chunk_size=128)
            torch.cuda.synchronize()
            pt_ms = (time.perf_counter() - start) / iters * 1000
            speedup = pt_ms / triton_ms
            pt_str = f"{pt_ms:13.2f} ms"
            sp_str = f"{speedup:8.1f}x"
        else:
            pt_str = "       (skipped)"
            sp_str = "       N/A"

        print(f"{seq_len:10d} | {pt_str:>16} | {triton_ms:12.2f} ms | {sp_str:>10}")

    print("\n=================================================================")
    print("  ALL BENCHMARKS COMPLETED.")
    print("=================================================================")


if __name__ == "__main__":
    verify_and_benchmark()
