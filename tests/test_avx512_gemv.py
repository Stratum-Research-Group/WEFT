import pytest
import torch
import numpy as np
import weft_avx512

def test_avx512_exact_mathematical_soundness():
    """Verify bit-exact soundness between AVX-512 popcount GEMV and naive bipolar dot product."""
    rows = 64
    cols = 512
    cols_u64 = cols // 64

    # Fix seed for deterministic reproduction
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate random bits {0, 1}
    b_bits = np.random.randint(0, 2, size=(rows, cols), dtype=np.uint8)
    x_bits = np.random.randint(0, 2, size=(cols,), dtype=np.uint8)

    # Convert to bipolar {-1, +1}
    # 1 -> +1, 0 -> -1
    b_bipolar = b_bits.astype(np.float32) * 2.0 - 1.0
    x_bipolar = x_bits.astype(np.float32) * 2.0 - 1.0

    # Ground truth mathematical dot product
    expected_dot = b_bipolar @ x_bipolar

    # 2. Pack bits into uint64 bitboards
    b_packed = np.zeros((rows, cols_u64), dtype=np.uint64)
    for r in range(rows):
        for c in range(cols_u64):
            word = 0
            for bit in range(64):
                if b_bits[r, c * 64 + bit] == 1:
                    word |= (1 << bit)
            b_packed[r, c] = word

    x_packed = np.zeros((cols_u64,), dtype=np.uint64)
    for c in range(cols_u64):
        word = 0
        for bit in range(64):
            if x_bits[c * 64 + bit] == 1:
                word |= (1 << bit)
        x_packed[c] = word

    # Convert to torch int64 (matching uint64 bit pattern)
    b_torch = torch.from_numpy(b_packed.view(np.int64))
    x_torch = torch.from_numpy(x_packed.view(np.int64))
    out_avx = torch.zeros(rows, dtype=torch.float32)

    scale = 1.0
    weft_avx512.subbit_gemv_avx512(b_torch, x_torch, out_avx, scale)

    # 3. Assert exact bit-level equality
    avx_res = out_avx.numpy()
    np.testing.assert_allclose(avx_res, expected_dot, atol=1e-5, rtol=1e-5)
    print("Exact mathematical soundness confirmed: AVX-512 popcount == bipolar dot product!")

def test_avx512_scaling():
    rows = 32
    cols_u64 = 8
    b_torch = torch.randint(0, 2**62, (rows, cols_u64), dtype=torch.int64)
    x_torch = torch.randint(0, 2**62, (cols_u64,), dtype=torch.int64)
    
    out1 = torch.zeros(rows, dtype=torch.float32)
    out2 = torch.zeros(rows, dtype=torch.float32)
    
    weft_avx512.subbit_gemv_avx512(b_torch, x_torch, out1, 1.0)
    weft_avx512.subbit_gemv_avx512(b_torch, x_torch, out2, 2.5)
    
    np.testing.assert_allclose(out1.numpy() * 2.5, out2.numpy(), atol=1e-4)

def test_avx512_batch4_mathematical_soundness():
    """Verify bit-exact soundness of batch-4 GEMM against individual GEMV passes."""
    rows = 128
    cols = 1024
    cols_u64 = cols // 64

    torch.manual_seed(123)
    bitboards = torch.randint(0, 2**62, (rows, cols_u64), dtype=torch.int64)
    x_batch = torch.randint(0, 2**62, (4, cols_u64), dtype=torch.int64)

    out_gemm = torch.zeros(4, rows, dtype=torch.float32)
    weft_avx512.subbit_gemm_avx512_batch4(bitboards, x_batch, out_gemm, 0.05)

    # Compute 4 individual GEMV passes
    out_individual = torch.zeros(4, rows, dtype=torch.float32)
    for b in range(4):
        weft_avx512.subbit_gemv_avx512(bitboards, x_batch[b], out_individual[b], 0.05)

    np.testing.assert_allclose(out_gemm.numpy(), out_individual.numpy(), atol=1e-5, rtol=1e-5)
    print("Exact batch-4 mathematical soundness confirmed: GEMM batch-4 == 4x individual GEMV!")

def test_avx512_fused_gemv_mathematical_soundness():
    """Verify bit-exact soundness of fused popcount + rank-1 SVD update against unfused sequence."""
    rows = 64
    cols = 512
    cols_u64 = cols // 64

    torch.manual_seed(999)
    bitboards = torch.randint(0, 2**62, (rows, cols_u64), dtype=torch.int64)
    x_quant = torch.randint(0, 2**62, (cols_u64,), dtype=torch.int64)
    u_proj = torch.randn(rows, dtype=torch.float32)
    v_dot_x = 3.14159

    scale = 0.05

    # 1. Fused pass
    out_fused = torch.zeros(rows, dtype=torch.float32)
    weft_avx512.subbit_fused_gemv_avx512(bitboards, x_quant, u_proj, v_dot_x, out_fused, scale)

    # 2. Unfused reference: popcount GEMV + rank-1 outer addition
    out_unfused = torch.zeros(rows, dtype=torch.float32)
    weft_avx512.subbit_gemv_avx512(bitboards, x_quant, out_unfused, scale)
    out_unfused += u_proj * v_dot_x

    np.testing.assert_allclose(out_fused.numpy(), out_unfused.numpy(), atol=1e-5, rtol=1e-5)
    print("Exact fused mathematical soundness confirmed: Fused == Unfused Popcount + Rank-1!")
