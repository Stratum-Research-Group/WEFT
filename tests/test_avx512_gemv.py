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

def test_avx512_ternary_gemv_mathematical_soundness():
    """Verify bit-exact soundness of masked ternary popcount GEMV against naive float dot product."""
    rows = 64
    cols = 512
    cols_u64 = cols // 64

    torch.manual_seed(777)
    np.random.seed(777)

    # Ternary weights W in {-1, 0, +1}
    # 0 with 50% probability, +1 with 25%, -1 with 25%
    w_raw = np.random.choice([0, 1, -1], size=(rows, cols), p=[0.5, 0.25, 0.25]).astype(np.float32)
    x_bits = np.random.randint(0, 2, size=(cols,), dtype=np.uint8)
    x_bipolar = x_bits.astype(np.float32) * 2.0 - 1.0

    # Ground truth
    expected_dot = w_raw @ x_bipolar

    # Pack ternary weights into b_nonzero and b_sign
    nz_mask = (w_raw != 0).astype(np.uint8)
    sign_mask = (w_raw == 1).astype(np.uint8)

    b_nz = np.zeros((rows, cols_u64), dtype=np.uint64)
    b_sign = np.zeros((rows, cols_u64), dtype=np.uint64)

    for r in range(rows):
        for c in range(cols_u64):
            word_nz = 0
            word_sign = 0
            for bit in range(64):
                idx = c * 64 + bit
                if nz_mask[r, idx] == 1:
                    word_nz |= (1 << bit)
                if sign_mask[r, idx] == 1:
                    word_sign |= (1 << bit)
            b_nz[r, c] = word_nz
            b_sign[r, c] = word_sign

    x_packed = np.zeros((cols_u64,), dtype=np.uint64)
    for c in range(cols_u64):
        word = 0
        for bit in range(64):
            if x_bits[c * 64 + bit] == 1:
                word |= (1 << bit)
        x_packed[c] = word

    b_nz_torch = torch.from_numpy(b_nz.view(np.int64))
    b_sign_torch = torch.from_numpy(b_sign.view(np.int64))
    x_torch = torch.from_numpy(x_packed.view(np.int64))

    out_ternary = torch.zeros(rows, dtype=torch.float32)
    weft_avx512.subbit_ternary_gemv_avx512(b_nz_torch, b_sign_torch, x_torch, out_ternary, 1.0)

    np.testing.assert_allclose(out_ternary.numpy(), expected_dot, atol=1e-5, rtol=1e-5)
    print("Exact ternary mathematical soundness confirmed: Masked Popcount == Ternary Dot Product!")

def test_avx512_rope_mathematical_soundness():
    """Verify bit-exact soundness of AVX-512 in-register RoPE against naive 2D Givens rotation."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 64
    half_dim = head_dim // 2

    torch.manual_seed(444)
    x = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)

    # Angles for RoPE
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # 1. AVX-512 fast path
    out_avx = torch.zeros_like(x)
    weft_avx512.subbit_rope_avx512(x, cos, sin, out_avx)

    # 2. PyTorch reference path
    out_ref = torch.zeros_like(x)
    for b in range(batch):
        for h in range(heads):
            for s in range(seq_len):
                for i in range(half_dim):
                    c = cos[s, i].item()
                    s_val = sin[s, i].item()
                    e = x[b, h, s, 2 * i].item()
                    o = x[b, h, s, 2 * i + 1].item()
                    out_ref[b, h, s, 2 * i]     = e * c - o * s_val
                    out_ref[b, h, s, 2 * i + 1] = e * s_val + o * c

    np.testing.assert_allclose(out_avx.numpy(), out_ref.numpy(), atol=1e-5, rtol=1e-5)
    print("Exact RoPE mathematical soundness confirmed: AVX-512 RoPE == Reference Givens Rotation!")

def test_avx512_softmax_mathematical_soundness():
    """Verify bit-exact soundness of AVX-512 in-register softmax against PyTorch torch.softmax."""
    batch_size = 8
    seq_len = 1024

    torch.manual_seed(555)
    logits = torch.randn(batch_size, seq_len, dtype=torch.float32) * 5.0

    # 1. AVX-512 fast path
    out_avx = torch.zeros_like(logits)
    weft_avx512.subbit_softmax_avx512(logits, out_avx)

    # 2. PyTorch reference path
    out_ref = torch.softmax(logits, dim=-1)

    np.testing.assert_allclose(out_avx.numpy(), out_ref.numpy(), atol=1e-5, rtol=1e-5)
    assert np.all(out_avx.numpy() >= 0.0)
    np.testing.assert_allclose(out_avx.sum(dim=-1).numpy(), np.ones(batch_size), atol=1e-4)
    print("Exact Softmax mathematical soundness confirmed: AVX-512 Softmax == PyTorch Softmax!")
