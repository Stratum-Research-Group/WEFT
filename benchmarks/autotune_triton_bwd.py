"""
Triton Fused Backward Autograd Kernel Autotuner for WEFT Recurrence.
Benchmarks across tile configurations (BLOCK_D, num_warps) on RTX 5060 Ti.
"""

import time
import torch
import triton
from weft.kernels.fused_autograd import _fused_chunk_recurrent_bwd_kernel


def benchmark_backward_autotune():
    device = "cuda"
    b, s, h, d, r = 2, 4096, 4, 128, 16
    dtype = torch.bfloat16

    x = torch.randn(b, s, h, d, device=device, dtype=dtype)
    kp = torch.randn(b, s, h, r, device=device, dtype=dtype)
    vp = torch.randn(b, s, h, r, device=device, dtype=dtype)
    gamma = torch.randn(h, r, device=device, dtype=dtype)
    dy = torch.randn(b, s, h, d, device=device, dtype=dtype)
    s_all = torch.randn(b, s, h, r, d, device=device, dtype=dtype)

    dx = torch.empty_like(x)
    dkp = torch.empty_like(kp)
    dvp = torch.empty_like(vp)

    configs = [
        (32, 2),
        (32, 4),
        (64, 2),
        (64, 4),
        (64, 8),
        (128, 4),
        (128, 8),
    ]

    print("=" * 60)
    print(f"  WEFT TRITON BACKWARD RECURRENT KERNEL AUTOTUNER")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Workload: B={b}, S={s}, H={h}, D={d}, R={r}, dtype={dtype}")
    print("=" * 60)
    print(f"{'BLOCK_D':>8} | {'num_warps':>10} | {'Latency (ms)':>14} | {'Tokens/sec':>12}")
    print("-" * 52)

    best_ms = float("inf")
    best_cfg = None

    for block_d, warps in configs:
        grid = (b, h, triton.cdiv(d, block_d))
        dgamma_partial = torch.zeros(b, h, triton.cdiv(d, block_d), r, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            _fused_chunk_recurrent_bwd_kernel[grid](
                x, kp, vp, gamma, dy, None, s_all,
                dx, dkp, dvp, dgamma_partial,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                kp.stride(0), kp.stride(1), kp.stride(2), kp.stride(3),
                vp.stride(0), vp.stride(1), vp.stride(2), vp.stride(3),
                gamma.stride(0), gamma.stride(1),
                dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
                0, 0, 0, 0,
                s_all.stride(0), s_all.stride(1), s_all.stride(2), s_all.stride(3), s_all.stride(4),
                dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
                dkp.stride(0), dkp.stride(1), dkp.stride(2), dkp.stride(3),
                dvp.stride(0), dvp.stride(1), dvp.stride(2), dvp.stride(3),
                dgamma_partial.stride(0), dgamma_partial.stride(1), dgamma_partial.stride(2), dgamma_partial.stride(3),
                B=b, S=s, H=h, D=d, R=r,
                BLOCK_D=block_d,
                HAS_DFINAL=False,
                num_warps=warps,
            )
        torch.cuda.synchronize()

        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            _fused_chunk_recurrent_bwd_kernel[grid](
                x, kp, vp, gamma, dy, None, s_all,
                dx, dkp, dvp, dgamma_partial,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                kp.stride(0), kp.stride(1), kp.stride(2), kp.stride(3),
                vp.stride(0), vp.stride(1), vp.stride(2), vp.stride(3),
                gamma.stride(0), gamma.stride(1),
                dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
                0, 0, 0, 0,
                s_all.stride(0), s_all.stride(1), s_all.stride(2), s_all.stride(3), s_all.stride(4),
                dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
                dkp.stride(0), dkp.stride(1), dkp.stride(2), dkp.stride(3),
                dvp.stride(0), dvp.stride(1), dvp.stride(2), dvp.stride(3),
                dgamma_partial.stride(0), dgamma_partial.stride(1), dgamma_partial.stride(2), dgamma_partial.stride(3),
                B=b, S=s, H=h, D=d, R=r,
                BLOCK_D=block_d,
                HAS_DFINAL=False,
                num_warps=warps,
            )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / iters * 1000.0
        tok_s = (b * s) / (latency_ms / 1000.0)

        print(f"{block_d:8d} | {warps:10d} | {latency_ms:12.2f} ms | {tok_s/1e6:10.2f}M")
        if latency_ms < best_ms:
            best_ms = latency_ms
            best_cfg = (block_d, warps)

    print("-" * 52)
    print(f"Optimal Backward configuration: BLOCK_D={best_cfg[0]}, num_warps={best_cfg[1]} ({best_ms:.2f} ms)")


if __name__ == "__main__":
    benchmark_backward_autotune()
