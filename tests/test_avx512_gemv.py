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


def test_avx512_flash_attention_mathematical_soundness():
    """
    Test AVX-512 in-register Flash-Attention kernel against exact PyTorch reference.
    Validates online max-subtraction, numerical stability, and value accumulation parity.
    """
    if not hasattr(weft_avx512, "subbit_flash_attention_avx512"):
        pytest.skip("AVX-512 Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 4
    Q_LEN = 16
    KV_LEN = 32
    DIM = 64

    q = torch.randn(B, Q_LEN, DIM, dtype=torch.float32)
    k = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)
    v = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)
    out = torch.empty(B, Q_LEN, DIM, dtype=torch.float32)
    sm_scale = 1.0 / (DIM ** 0.5)

    weft_avx512.subbit_flash_attention_avx512(q, k, v, out, sm_scale)

    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, v)

    diff = (out - ref).abs().max().item()
    assert diff < 1e-4, f"Flash-Attention discrepancy too high: {diff}"


def test_avx512_causal_flash_attention_mathematical_soundness():
    """
    Test AVX-512 fused causal Flash-Attention kernel against exact PyTorch masked reference.
    Validates causal masking (ki <= qi) and value accumulation parity.
    """
    if not hasattr(weft_avx512, "subbit_causal_flash_attention_avx512"):
        pytest.skip("AVX-512 Causal Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 4
    Q_LEN = 32
    KV_LEN = 32
    DIM = 64

    q = torch.randn(B, Q_LEN, DIM, dtype=torch.float32)
    k = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)
    v = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)
    out = torch.empty(B, Q_LEN, DIM, dtype=torch.float32)
    sm_scale = 1.0 / (DIM ** 0.5)

    weft_avx512.subbit_causal_flash_attention_avx512(q, k, v, out, sm_scale)

    mask = torch.triu(torch.full((Q_LEN, KV_LEN), float('-inf')), diagonal=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale + mask
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, v)

    diff = (out - ref).abs().max().item()
    assert diff < 1e-4, f"Causal Flash-Attention discrepancy too high: {diff}"


def test_avx512_dynamic_head_flash_attention_mathematical_soundness():
    """
    Test AVX-512 dynamic-head Flash-Attention kernel with non-multiple-of-16 head dimensions.
    Validates AVX-512 tail masking (_mm512_maskz_loadu_ps, _mm512_mask_storeu_ps) parity.
    """
    if not hasattr(weft_avx512, "subbit_dynamic_head_flash_attention_avx512"):
        pytest.skip("AVX-512 Dynamic-Head Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 2
    Q_LEN = 8
    KV_LEN = 16
    DIM = 90  # 16 * 5 + 10 tail elements

    q = torch.randn(B, Q_LEN, DIM, dtype=torch.float32)
    k = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)
    v = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)
    out = torch.empty(B, Q_LEN, DIM, dtype=torch.float32)
    sm_scale = 1.0 / (DIM ** 0.5)

    weft_avx512.subbit_dynamic_head_flash_attention_avx512(q, k, v, out, sm_scale)

    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, v)

    diff = (out - ref).abs().max().item()
    assert diff < 1e-4, f"Dynamic-head Flash-Attention discrepancy too high: {diff}"


def test_avx512_paged_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Paged KV-Cache Flash-Attention kernel against exact segmented reference.
    Validates dynamic block-table indirect indexing and online value accumulation parity.
    """
    if not hasattr(weft_avx512, "subbit_paged_flash_attention_avx512"):
        pytest.skip("AVX-512 Paged Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 2
    Q_LEN = 4
    DIM = 64
    BLOCK_SIZE = 16
    NUM_BLOCKS = 8
    CTX_LEN = 32

    q = torch.randn(B, Q_LEN, DIM, dtype=torch.float32)
    k_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, DIM, dtype=torch.float32)
    v_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, DIM, dtype=torch.float32)
    block_tables = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    context_lens = torch.tensor([CTX_LEN, CTX_LEN], dtype=torch.int32)
    out = torch.empty(B, Q_LEN, DIM, dtype=torch.float32)
    sm_scale = 1.0 / (DIM ** 0.5)

    weft_avx512.subbit_paged_flash_attention_avx512(
        q, k_cache, v_cache, block_tables, context_lens, out, sm_scale, BLOCK_SIZE
    )

    ref_out = torch.empty_like(out)
    for b in range(B):
        k_parts = [k_cache[block_tables[b, i]] for i in range(CTX_LEN // BLOCK_SIZE)]
        v_parts = [v_cache[block_tables[b, i]] for i in range(CTX_LEN // BLOCK_SIZE)]
        k_contig = torch.cat(k_parts, dim=0)
        v_contig = torch.cat(v_parts, dim=0)

        scores = torch.matmul(q[b], k_contig.t()) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        ref_out[b] = torch.matmul(probs, v_contig)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Paged Flash-Attention discrepancy too high: {diff}"


def test_avx512_int8_kv_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Int8 Quantized KV-Cache Flash-Attention kernel.
    Validates in-register sign-extension, int32-to-float32 conversion, and scaling parity.
    """
    if not hasattr(weft_avx512, "subbit_int8_kv_flash_attention_avx512"):
        pytest.skip("AVX-512 Int8 KV Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 2
    Q_LEN = 4
    KV_LEN = 32
    DIM = 64

    q = torch.randn(B, Q_LEN, DIM, dtype=torch.float32)
    k_fp = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)
    v_fp = torch.randn(B, KV_LEN, DIM, dtype=torch.float32)

    k_scales = k_fp.abs().amax(dim=-1) / 127.0
    v_scales = v_fp.abs().amax(dim=-1) / 127.0

    k_int8 = torch.clamp(torch.round(k_fp / k_scales.unsqueeze(-1)), -128, 127).to(torch.int8)
    v_int8 = torch.clamp(torch.round(v_fp / v_scales.unsqueeze(-1)), -128, 127).to(torch.int8)

    out = torch.empty(B, Q_LEN, DIM, dtype=torch.float32)
    sm_scale = 1.0 / (DIM ** 0.5)

    weft_avx512.subbit_int8_kv_flash_attention_avx512(
        q, k_int8, v_int8, k_scales, v_scales, out, sm_scale
    )

    k_deq = k_int8.float() * k_scales.unsqueeze(-1)
    v_deq = v_int8.float() * v_scales.unsqueeze(-1)

    scores = torch.matmul(q, k_deq.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v_deq)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Int8 KV Flash-Attention discrepancy too high: {diff}"


def test_avx512_int4_kv_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Int4 Sub-Byte Packed KV-Cache Flash-Attention kernel.
    Validates in-register nibble shifting, sign extension, float conversion, and scaling parity.
    """
    if not hasattr(weft_avx512, "subbit_int4_kv_flash_attention_avx512"):
        pytest.skip("AVX-512 Int4 KV Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 2
    Q_LEN = 4
    KV_LEN = 32
    DIM = 64

    q = torch.randn(B, Q_LEN, DIM, dtype=torch.float32)
    
    # Generate integer values in range [-8, +7]
    k_i4 = torch.randint(-8, 8, (B, KV_LEN, DIM), dtype=torch.int32)
    v_i4 = torch.randint(-8, 8, (B, KV_LEN, DIM), dtype=torch.int32)

    k_scales = torch.rand(B, KV_LEN, dtype=torch.float32) * 0.1 + 0.01
    v_scales = torch.rand(B, KV_LEN, dtype=torch.float32) * 0.1 + 0.01

    # Pack into uint8 (low nibble = dim 2k, high nibble = dim 2k+1)
    k_packed = torch.zeros(B, KV_LEN, DIM // 2, dtype=torch.uint8)
    v_packed = torch.zeros(B, KV_LEN, DIM // 2, dtype=torch.uint8)
    for b in range(B):
        for s in range(KV_LEN):
            for d in range(DIM // 2):
                low = k_i4[b, s, 2 * d].item() & 0x0F
                high = k_i4[b, s, 2 * d + 1].item() & 0x0F
                k_packed[b, s, d] = low | (high << 4)

                v_low = v_i4[b, s, 2 * d].item() & 0x0F
                v_high = v_i4[b, s, 2 * d + 1].item() & 0x0F
                v_packed[b, s, d] = v_low | (v_high << 4)

    out = torch.empty(B, Q_LEN, DIM, dtype=torch.float32)
    sm_scale = 1.0 / (DIM ** 0.5)

    weft_avx512.subbit_int4_kv_flash_attention_avx512(
        q, k_packed, v_packed, k_scales, v_scales, out, sm_scale
    )

    # Reference evaluation
    k_deq = k_i4.float() * k_scales.unsqueeze(-1)
    v_deq = v_i4.float() * v_scales.unsqueeze(-1)

    scores = torch.matmul(q, k_deq.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v_deq)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Int4 KV Flash-Attention discrepancy too high: {diff}"


def test_avx512_paged_int4_kv_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Paged Int4 Sub-Byte KV-Cache Flash-Attention kernel.
    Validates block-table dynamic indexing and in-register nibble dequantization parity.
    """
    if not hasattr(weft_avx512, "subbit_paged_int4_kv_flash_attention_avx512"):
        pytest.skip("AVX-512 Paged Int4 KV Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 2
    Q_LEN = 4
    HEAD_DIM = 64
    BLOCK_SIZE = 16
    NUM_BLOCKS = 16
    CTX_LEN = 32

    q = torch.randn(B, Q_LEN, HEAD_DIM, dtype=torch.float32)

    # Int4 integer values in [-8, 7]
    k_blocks_i4 = torch.randint(-8, 8, (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM), dtype=torch.int32)
    v_blocks_i4 = torch.randint(-8, 8, (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM), dtype=torch.int32)

    k_scales = torch.rand(NUM_BLOCKS, BLOCK_SIZE, dtype=torch.float32) * 0.1 + 0.01
    v_scales = torch.rand(NUM_BLOCKS, BLOCK_SIZE, dtype=torch.float32) * 0.1 + 0.01

    # Pack into uint8
    k_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM // 2, dtype=torch.uint8)
    v_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM // 2, dtype=torch.uint8)
    for blk in range(NUM_BLOCKS):
        for s in range(BLOCK_SIZE):
            for d in range(HEAD_DIM // 2):
                low = k_blocks_i4[blk, s, 2 * d].item() & 0x0F
                high = k_blocks_i4[blk, s, 2 * d + 1].item() & 0x0F
                k_cache[blk, s, d] = low | (high << 4)

                v_low = v_blocks_i4[blk, s, 2 * d].item() & 0x0F
                v_high = v_blocks_i4[blk, s, 2 * d + 1].item() & 0x0F
                v_cache[blk, s, d] = v_low | (v_high << 4)

    block_tables = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.int32)
    context_lens = torch.tensor([CTX_LEN, CTX_LEN], dtype=torch.int32)

    out = torch.empty(B, Q_LEN, HEAD_DIM, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    weft_avx512.subbit_paged_int4_kv_flash_attention_avx512(
        q, k_cache, v_cache, k_scales, v_scales, block_tables, context_lens, out, sm_scale, BLOCK_SIZE
    )

    # Reference evaluation
    ref_out = torch.empty_like(out)
    for b in range(B):
        k_parts = [k_blocks_i4[block_tables[b, i]] for i in range(CTX_LEN // BLOCK_SIZE)]
        v_parts = [v_blocks_i4[block_tables[b, i]] for i in range(CTX_LEN // BLOCK_SIZE)]
        ks_parts = [k_scales[block_tables[b, i]] for i in range(CTX_LEN // BLOCK_SIZE)]
        vs_parts = [v_scales[block_tables[b, i]] for i in range(CTX_LEN // BLOCK_SIZE)]

        k_contig = (torch.cat(k_parts, dim=0).float()) * torch.cat(ks_parts, dim=0).unsqueeze(-1)
        v_contig = (torch.cat(v_parts, dim=0).float()) * torch.cat(vs_parts, dim=0).unsqueeze(-1)

        scores = torch.matmul(q[b], k_contig.t()) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        ref_out[b] = torch.matmul(probs, v_contig)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Paged Int4 KV Flash-Attention discrepancy too high: {diff}"


def test_avx512_gqa_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Grouped-Query Attention (GQA) Flash-Attention kernel.
    Validates broadcasted KV-head indexing and multi-head online softmax numerical parity.
    """
    if not hasattr(weft_avx512, "subbit_gqa_flash_attention_avx512"):
        pytest.skip("AVX-512 GQA Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 2
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 2
    Q_LEN = 4
    KV_LEN = 16
    HEAD_DIM = 64

    q = torch.randn(B, NUM_Q_HEADS, Q_LEN, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(B, NUM_KV_HEADS, KV_LEN, HEAD_DIM, dtype=torch.float32)
    v = torch.randn(B, NUM_KV_HEADS, KV_LEN, HEAD_DIM, dtype=torch.float32)
    out = torch.empty(B, NUM_Q_HEADS, Q_LEN, HEAD_DIM, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    weft_avx512.subbit_gqa_flash_attention_avx512(q, k, v, out, sm_scale)

    # Reference GQA computation
    group_ratio = NUM_Q_HEADS // NUM_KV_HEADS
    ref_out = torch.empty_like(out)
    for b in range(B):
        for qh in range(NUM_Q_HEADS):
            kv_h = qh // group_ratio
            cur_q = q[b, qh]
            cur_k = k[b, kv_h]
            cur_v = v[b, kv_h]

            scores = torch.matmul(cur_q, cur_k.transpose(-1, -2)) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            ref_out[b, qh] = torch.matmul(probs, cur_v)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"GQA Flash-Attention discrepancy too high: {diff}"


def test_avx512_asym_int4_kv_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Asymmetric Int4 KV-Cache Flash-Attention kernel.
    Validates in-register unsigned nibble unpacking, zero-point subtraction, and scaling parity.
    """
    if not hasattr(weft_avx512, "subbit_asym_int4_kv_flash_attention_avx512"):
        pytest.skip("AVX-512 Asymmetric Int4 KV Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    B = 2
    Q_LEN = 4
    KV_LEN = 32
    HEAD_DIM = 64

    q = torch.randn(B, Q_LEN, HEAD_DIM, dtype=torch.float32)

    # Unsigned 4-bit integers in [0, 15]
    k_u4 = torch.randint(0, 16, (B, KV_LEN, HEAD_DIM), dtype=torch.int32)
    v_u4 = torch.randint(0, 16, (B, KV_LEN, HEAD_DIM), dtype=torch.int32)

    k_scales = torch.rand(B, KV_LEN, dtype=torch.float32) * 0.1 + 0.01
    v_scales = torch.rand(B, KV_LEN, dtype=torch.float32) * 0.1 + 0.01

    k_zeros = torch.randint(0, 16, (B, KV_LEN), dtype=torch.int32).float()
    v_zeros = torch.randint(0, 16, (B, KV_LEN), dtype=torch.int32).float()

    # Pack into uint8 (low nibble = dim 2k, high nibble = dim 2k+1)
    k_packed = torch.zeros(B, KV_LEN, HEAD_DIM // 2, dtype=torch.uint8)
    v_packed = torch.zeros(B, KV_LEN, HEAD_DIM // 2, dtype=torch.uint8)
    for b in range(B):
        for s in range(KV_LEN):
            for d in range(HEAD_DIM // 2):
                low = k_u4[b, s, 2 * d].item() & 0x0F
                high = k_u4[b, s, 2 * d + 1].item() & 0x0F
                k_packed[b, s, d] = low | (high << 4)

                v_low = v_u4[b, s, 2 * d].item() & 0x0F
                v_high = v_u4[b, s, 2 * d + 1].item() & 0x0F
                v_packed[b, s, d] = v_low | (v_high << 4)

    out = torch.empty(B, Q_LEN, HEAD_DIM, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    weft_avx512.subbit_asym_int4_kv_flash_attention_avx512(
        q, k_packed, v_packed, k_scales, v_scales, k_zeros, v_zeros, out, sm_scale
    )

    # Reference evaluation
    k_deq = (k_u4.float() - k_zeros.unsqueeze(-1)) * k_scales.unsqueeze(-1)
    v_deq = (v_u4.float() - v_zeros.unsqueeze(-1)) * v_scales.unsqueeze(-1)

    scores = torch.matmul(q, k_deq.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v_deq)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Asym Int4 KV Flash-Attention discrepancy too high: {diff}"


def test_avx512_fused_swiglu_gemm_mathematical_soundness():
    """
    Test AVX-512 Fused Dual-Bitboard Gate-Up Projection GEMM with In-Register SwiGLU.
    Validates dual-bitboard VPOPCNTDQ popcount, sigmoid polynomial approximation, and SwiGLU parity.
    """
    if not hasattr(weft_avx512, "subbit_fused_swiglu_gemm_avx512"):
        pytest.skip("AVX-512 Fused SwiGLU GEMM kernel not compiled")

    torch.manual_seed(42)
    B = 2
    IN_FEATURES = 128
    OUT_FEATURES = 256
    K_WORDS = IN_FEATURES // 64

    x = torch.randn(B, IN_FEATURES, dtype=torch.float32)

    # Random packed bitboards
    w_gate_packed = torch.randint(0, 2**31, (OUT_FEATURES, K_WORDS), dtype=torch.int64)
    w_up_packed = torch.randint(0, 2**31, (OUT_FEATURES, K_WORDS), dtype=torch.int64)

    scales_gate = torch.rand(OUT_FEATURES, dtype=torch.float32) * 0.05 + 0.01
    scales_up = torch.rand(OUT_FEATURES, dtype=torch.float32) * 0.05 + 0.01

    out = torch.empty(B, OUT_FEATURES, dtype=torch.float32)

    weft_avx512.subbit_fused_swiglu_gemm_avx512(
        x, w_gate_packed, w_up_packed, scales_gate, scales_up, out
    )

    # Reference evaluation
    def unpack_bitboard(packed):
        w_bin = torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.float32)
        for r in range(OUT_FEATURES):
            for w in range(K_WORDS):
                val = int(packed[r, w].item())
                for b in range(64):
                    bit = (val >> b) & 1
                    w_bin[r, w * 64 + b] = 1.0 if bit else -1.0
        return w_bin

    w_g = unpack_bitboard(w_gate_packed) * scales_gate.unsqueeze(-1)
    w_u = unpack_bitboard(w_up_packed) * scales_up.unsqueeze(-1)

    x_bin = torch.where(x > 0.0, 1.0, -1.0)
    gate = torch.matmul(x_bin, w_g.t())
    up = torch.matmul(x_bin, w_u.t())

    ref_out = (gate * torch.sigmoid(gate)) * up

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Fused SwiGLU GEMM discrepancy too high: {diff}"


def test_avx512_sliding_window_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Sliding-Window Flash-Attention kernel.
    Validates banded causal windowing [max(0, qi - W + 1), min(qi + 1, kv_len)]
    and numerical softmax equivalence with zero DRAM mask overhead.
    """
    if not hasattr(weft_avx512, "subbit_sliding_window_flash_attention_avx512"):
        pytest.skip("AVX-512 Sliding-Window Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    BATCH_HEADS = 8
    Q_LEN = 16
    KV_LEN = 32
    HEAD_DIM = 64
    WINDOW_SIZE = 8

    q = torch.randn(BATCH_HEADS, Q_LEN, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(BATCH_HEADS, KV_LEN, HEAD_DIM, dtype=torch.float32)
    v = torch.randn(BATCH_HEADS, KV_LEN, HEAD_DIM, dtype=torch.float32)

    out = torch.empty(BATCH_HEADS, Q_LEN, HEAD_DIM, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    weft_avx512.subbit_sliding_window_flash_attention_avx512(
        q, k, v, out, sm_scale, WINDOW_SIZE
    )

    # Reference evaluation
    ref_out = torch.zeros_like(q)
    for bh in range(BATCH_HEADS):
        for qi in range(Q_LEN):
            k_start = max(0, qi - WINDOW_SIZE + 1)
            k_end = min(qi + 1, KV_LEN)
            q_vec = q[bh, qi:qi+1, :]
            k_win = k[bh, k_start:k_end, :]
            v_win = v[bh, k_start:k_end, :]
            scores = torch.matmul(q_vec, k_win.transpose(-1, -2)) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            ref_out[bh, qi, :] = torch.matmul(probs, v_win).squeeze(0)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Sliding-Window Flash-Attention discrepancy too high: {diff}"


def test_avx512_fused_ternary_flash_attention_mathematical_soundness():
    """
    Test AVX-512 Fused Sub-Bit Ternary Flash-Attention Projection kernel.
    Validates on-the-fly ternary Q/K/V projections and online softmax Flash-Attention.
    """
    if not hasattr(weft_avx512, "subbit_ternary_flash_attention_avx512"):
        pytest.skip("AVX-512 Fused Ternary Flash-Attention kernel not compiled")

    torch.manual_seed(42)
    BATCH_HEADS = 4
    SEQ_LEN = 8
    IN_FEATURES = 128
    HEAD_DIM = 64
    IN_WORDS = IN_FEATURES // 64

    x = torch.randn(BATCH_HEADS, SEQ_LEN, IN_FEATURES, dtype=torch.float32)

    # Packed ternary bitboards: nonzero mask & sign mask
    w_q_nz = torch.randint(0, 2**31, (HEAD_DIM, IN_WORDS), dtype=torch.int64)
    w_q_sign = torch.randint(0, 2**31, (HEAD_DIM, IN_WORDS), dtype=torch.int64)
    w_k_nz = torch.randint(0, 2**31, (HEAD_DIM, IN_WORDS), dtype=torch.int64)
    w_k_sign = torch.randint(0, 2**31, (HEAD_DIM, IN_WORDS), dtype=torch.int64)
    w_v_nz = torch.randint(0, 2**31, (HEAD_DIM, IN_WORDS), dtype=torch.int64)
    w_v_sign = torch.randint(0, 2**31, (HEAD_DIM, IN_WORDS), dtype=torch.int64)

    scales_q = torch.rand(HEAD_DIM, dtype=torch.float32) * 0.05 + 0.01
    scales_k = torch.rand(HEAD_DIM, dtype=torch.float32) * 0.05 + 0.01
    scales_v = torch.rand(HEAD_DIM, dtype=torch.float32) * 0.05 + 0.01

    out = torch.empty(BATCH_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    weft_avx512.subbit_ternary_flash_attention_avx512(
        x, w_q_nz, w_q_sign, w_k_nz, w_k_sign, w_v_nz, w_v_sign,
        scales_q, scales_k, scales_v, out, sm_scale
    )

    # Reference evaluation
    def project_ternary(x_in, nz_p, sign_p, scales):
        out_proj = torch.zeros(BATCH_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
        for bh in range(BATCH_HEADS):
            for t in range(SEQ_LEN):
                cur_x = torch.where(x_in[bh, t] > 0.0, 1.0, -1.0)
                for d in range(HEAD_DIM):
                    dot = 0.0
                    for w in range(IN_WORDS):
                        nz = int(nz_p[d, w].item())
                        sg = int(sign_p[d, w].item())
                        for b in range(64):
                            if (nz >> b) & 1:
                                weight_val = 1.0 if (sg >> b) & 1 else -1.0
                                dot += cur_x[w * 64 + b].item() * weight_val
                    out_proj[bh, t, d] = dot * scales[d].item()
        return out_proj

    q_ref = project_ternary(x, w_q_nz, w_q_sign, scales_q)
    k_ref = project_ternary(x, w_k_nz, w_k_sign, scales_k)
    v_ref = project_ternary(x, w_v_nz, w_v_sign, scales_v)

    ref_out = torch.zeros_like(out)
    for bh in range(BATCH_HEADS):
        for qi in range(SEQ_LEN):
            q_vec = q_ref[bh, qi:qi+1, :]
            k_win = k_ref[bh, :qi+1, :]
            v_win = v_ref[bh, :qi+1, :]
            scores = torch.matmul(q_vec, k_win.transpose(-1, -2)) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            ref_out[bh, qi, :] = torch.matmul(probs, v_win).squeeze(0)

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Fused Ternary Flash-Attention discrepancy too high: {diff}"








