import time
import torch
import numpy as np
import weft_avx512

def benchmark_avx512_throughput():
    print("=" * 80)
    print("  WEFT NATIVE AVX-512 VPOPCNTDQ HARDWARE PROFILING SUITE")
    print("  Target: AMD Ryzen 5 7600X 6-Core (Zen 4 AVX-512 Architecture)")
    print("=" * 80)

    # Test real LLM projection dimensions (e.g. Llama-3.3-70B, Qwen-72B, Mistral-24B)
    matrix_shapes = [
        (4096, 4096, "Mistral-Small-24B (Q/K/V Attention)"),
        (14336, 4096, "Mistral-Small-24B (MLP Gate/Up Projection)"),
        (8192, 8192, "Llama-3.3-70B (Q/K/V Attention)"),
        (28672, 8192, "Llama-3.3-70B (MLP Up Projection)")
    ]

    results = []

    for rows, cols, desc in matrix_shapes:
        cols_u64 = cols // 64
        # Allocate tensors
        bitboards = torch.randint(0, 2**62, (rows, cols_u64), dtype=torch.int64)
        x_quant = torch.randint(0, 2**62, (cols_u64,), dtype=torch.int64)
        out = torch.zeros(rows, dtype=torch.float32)

        # Baseline PyTorch FP32 matrix-vector multiplication
        x_fp32 = torch.randn(cols, dtype=torch.float32)
        w_fp32 = torch.randn(rows, cols, dtype=torch.float32)

        # Warmup
        for _ in range(5):
            weft_avx512.subbit_gemv_avx512(bitboards, x_quant, out, 0.05)
            _ = torch.mv(w_fp32, x_fp32)

        # Benchmark AVX-512 VPOPCNTDQ Sub-Bit GEMV
        iters = 50 if rows <= 8192 else 20
        t0 = time.perf_counter()
        for _ in range(iters):
            weft_avx512.subbit_gemv_avx512(bitboards, x_quant, out, 0.05)
        t_avx = (time.perf_counter() - t0) / iters

        # Benchmark PyTorch FP32 GEMV
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = torch.mv(w_fp32, x_fp32)
        t_torch = (time.perf_counter() - t0) / iters

        # Total operations: 2 * rows * cols FLOPs equivalent
        ops = 2.0 * rows * cols
        gflops_avx = (ops / t_avx) / 1e9
        gflops_torch = (ops / t_torch) / 1e9
        speedup = t_torch / t_avx

        res = {
            "desc": desc,
            "shape": f"{rows}x{cols}",
            "avx512_latency_us": round(t_avx * 1e6, 2),
            "torch_fp32_latency_us": round(t_torch * 1e6, 2),
            "avx512_effective_gflops": round(gflops_avx, 2),
            "speedup_vs_fp32": round(speedup, 2)
        }
        results.append(res)

        print(f"\n[Matrix: {desc} ({rows}x{cols})]")
        print(f"  AVX-512 Sub-Bit Latency : {t_avx * 1e6:8.2f} µs ({gflops_avx:8.2f} GFLOPs/s equiv)")
        print(f"  PyTorch FP32 Latency    : {t_torch * 1e6:8.2f} µs ({gflops_torch:8.2f} GFLOPs/s)")
        print(f"  Hardware Speedup Factor : {speedup:8.2f}x")

    print("\n" + "=" * 80)
    print("  HARDWARE PROFILING COMPLETED")
    print("=" * 80)
    return results

if __name__ == "__main__":
    benchmark_avx512_throughput()

def benchmark_batch4_gemm():
    print("\n" + "=" * 80)
    print("  BATCH-4 MULTI-TOKEN REGISTER REUSE BENCHMARK")
    print("=" * 80)

    rows, cols = 14336, 4096
    cols_u64 = cols // 64

    bitboards = torch.randint(0, 2**62, (rows, cols_u64), dtype=torch.int64)
    x_batch = torch.randint(0, 2**62, (4, cols_u64), dtype=torch.int64)
    out_batch = torch.zeros(4, rows, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        weft_avx512.subbit_gemm_avx512_batch4(bitboards, x_batch, out_batch, 0.05)

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        weft_avx512.subbit_gemm_avx512_batch4(bitboards, x_batch, out_batch, 0.05)
    t_batch = (time.perf_counter() - t0) / iters

    total_tokens = 4
    total_ops = 2.0 * rows * cols * total_tokens
    effective_gflops = (total_ops / t_batch) / 1e9

    print(f"Matrix: Mistral-Small-24B MLP ({rows}x{cols}) | Batch: 4 Tokens")
    print(f"  Batch-4 Execution Time : {t_batch * 1e6:8.2f} µs ({t_batch * 1e6 / 4:8.2f} µs/token)")
    print(f"  Effective Throughput   : {effective_gflops:8.2f} GFLOPs/s equivalent")

if __name__ == "__main__":
    benchmark_batch4_gemm()

def benchmark_fused_gemv():
    print("\n" + "=" * 80)
    print("  FUSED AVX-512 VPOPCNTDQ + RANK-1 SVD LOW-RANK CORRECTION BENCHMARK")
    print("=" * 80)

    rows, cols = 8192, 8192
    cols_u64 = cols // 64

    bitboards = torch.randint(0, 2**62, (rows, cols_u64), dtype=torch.int64)
    x_quant = torch.randint(0, 2**62, (cols_u64,), dtype=torch.int64)
    u_proj = torch.randn(rows, dtype=torch.float32)
    v_dot_x = 1.61803

    out_fused = torch.zeros(rows, dtype=torch.float32)
    out_unfused = torch.zeros(rows, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        weft_avx512.subbit_fused_gemv_avx512(bitboards, x_quant, u_proj, v_dot_x, out_fused, 0.05)
        weft_avx512.subbit_gemv_avx512(bitboards, x_quant, out_unfused, 0.05)
        out_unfused += u_proj * v_dot_x

    iters = 20
    # 1. Benchmark Unfused
    t0 = time.perf_counter()
    for _ in range(iters):
        weft_avx512.subbit_gemv_avx512(bitboards, x_quant, out_unfused, 0.05)
        out_unfused += u_proj * v_dot_x
    t_unfused = (time.perf_counter() - t0) / iters

    # 2. Benchmark Fused
    t0 = time.perf_counter()
    for _ in range(iters):
        weft_avx512.subbit_fused_gemv_avx512(bitboards, x_quant, u_proj, v_dot_x, out_fused, 0.05)
    t_fused = (time.perf_counter() - t0) / iters

    print(f"Matrix: Llama-3.3-70B Attention Projection ({rows}x{cols})")
    print(f"  Unfused Latency (GEMV + Add) : {t_unfused * 1e6:8.2f} µs")
    print(f"  Fused Latency (Single Pass)  : {t_fused * 1e6:8.2f} µs")
    print(f"  Fused Speedup Factor         : {t_unfused / t_fused:8.2f}x (Zero intermediate memory traffic)")

if __name__ == "__main__":
    benchmark_fused_gemv()
