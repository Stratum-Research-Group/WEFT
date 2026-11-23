"""
High-performance CPU bitboard unpacking and popcount-accelerated GEMM for .weft sub-bit models.
Optimized for AVX-512 VPOPCNTDQ (Zen 4 / Sapphire Rapids) with fallback to PyTorch vectorization.
Unifies 1-bit popcount, batch-4 register-reused S-GEMM, fused rank-1 SVD, masked ternary, and RoPE.
"""

import torch
import numpy as np

# Dynamically import native AVX-512 hardware accelerator if available on host
_AVX512_AVAILABLE = False
weft_avx512 = None
try:
    import weft_avx512
    _AVX512_AVAILABLE = True
except ImportError:
    try:
        import os
        from torch.utils.cpp_extension import load
        build_so = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build", "libweft_avx512.so"))
        if os.path.exists(build_so):
            import importlib.util
            spec = importlib.util.spec_from_file_location("weft_avx512", build_so)
            weft_avx512 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(weft_avx512)
            _AVX512_AVAILABLE = True
    except Exception:
        _AVX512_AVAILABLE = False


def is_avx512_accelerated() -> bool:
    """Returns True if native AVX-512 VPOPCNTDQ acceleration is active."""
    return _AVX512_AVAILABLE


def pack_activations_to_u64(x_sign: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of signs [..., rank] (with values 0 or 1) into packed uint64/int64 words.
    Each word holds 64 boolean signs packed along bit positions 0..63.
    """
    orig_shape = x_sign.shape
    rank = orig_shape[-1]
    num_words = (rank + 63) // 64
    pad_len = num_words * 64 - rank
    if pad_len > 0:
        x_pad = torch.nn.functional.pad(x_sign, (0, pad_len), value=0)
    else:
        x_pad = x_sign
    x_pad = x_pad.view(-1, num_words, 64).to(torch.int64)
    shifts = torch.arange(64, dtype=torch.int64, device=x_sign.device)
    packed = (x_pad << shifts).sum(dim=-1)
    return packed.view(*orig_shape[:-1], num_words)


def cpu_unpack_bitboards(packed: torch.Tensor, rank: int, mask_nonzero: torch.Tensor = None) -> torch.Tensor:
    """
    Vectorized unpacking of packed int32/int64 bitboards into bipolar {-1.0, +1.0} or ternary {-1.0, 0.0, +1.0} float32 tensors.
    Correctly recognizes 64-bit int64 vs 32-bit int32 storage and applies non-zero ternary masks.
    """
    N, num_words = packed.shape
    is_64 = (packed.dtype == torch.int64)
    word_bits = 64 if is_64 else 32
    shift_dtype = torch.int64 if is_64 else torch.int32
    shifts = torch.arange(word_bits, dtype=shift_dtype, device=packed.device)
    
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bipolar = (bits.float() * 2.0 - 1.0).view(N, num_words * word_bits)[:, :rank]
    
    if mask_nonzero is not None:
        mask_bits = (mask_nonzero.unsqueeze(-1) >> shifts) & 1
        mask_f = mask_bits.float().view(N, num_words * word_bits)[:, :rank]
        bipolar = bipolar * mask_f
        
    return bipolar


def cpu_subbit_matmul(
    x: torch.Tensor,
    b_packed: torch.Tensor,
    rank: int,
    scale: float = 1.0,
    u_proj: torch.Tensor = None,
    v_proj: torch.Tensor = None,
    b_nonzero: torch.Tensor = None
) -> torch.Tensor:
    """
    Unified CPU Sub-Bit Dispatcher:
    Automatically routes across 1-bit GEMV, batch-4 GEMM, fused rank-1 SVD, or masked ternary.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, rank)
    batch_size = x_2d.shape[0]

    if _AVX512_AVAILABLE and b_packed.dtype == torch.int64:
        # Vectorized bit-packing of activation signs: [batch_size, (rank + 63) // 64] uint64
        x_sign = (x_2d > 0).to(torch.int64)
        x_packed = pack_activations_to_u64(x_sign)

        # Mode 1: Masked Ternary GEMV
        if b_nonzero is not None and hasattr(weft_avx512, "subbit_ternary_gemv_avx512") and batch_size == 1:
            out = torch.zeros(b_packed.shape[0], dtype=torch.float32, device=x.device)
            weft_avx512.subbit_ternary_gemv_avx512(b_nonzero, b_packed, x_packed.squeeze(0), out, scale)
            return out.view(*orig_shape[:-1], b_packed.shape[0])

        # Mode 2: Fused Rank-1 SVD GEMV
        if u_proj is not None and v_proj is not None and hasattr(weft_avx512, "subbit_fused_gemv_avx512") and batch_size == 1:
            v_dot_x = float(torch.dot(v_proj.float(), x_2d.squeeze(0).float()))
            out = torch.zeros(b_packed.shape[0], dtype=torch.float32, device=x.device)
            weft_avx512.subbit_fused_gemv_avx512(b_packed, x_packed.squeeze(0), u_proj.float(), v_dot_x, out, scale)
            return out.view(*orig_shape[:-1], b_packed.shape[0])

        # Mode 3: Batch-4 GEMM with register reuse
        if batch_size == 4 and hasattr(weft_avx512, "subbit_gemm_avx512_batch4"):
            out = torch.zeros(4, b_packed.shape[0], dtype=torch.float32, device=x.device)
            weft_avx512.subbit_gemm_avx512_batch4(b_packed, x_packed, out, scale)
            return out.view(*orig_shape[:-1], b_packed.shape[0])

        # Mode 4: Standard Single-Token GEMV
        if batch_size == 1 and hasattr(weft_avx512, "subbit_gemv_avx512"):
            out = torch.zeros(b_packed.shape[0], dtype=torch.float32, device=x.device)
            weft_avx512.subbit_gemv_avx512(b_packed, x_packed.squeeze(0), out, scale)
            return out.view(*orig_shape[:-1], b_packed.shape[0])

    # Fallback Path
    bipolar = cpu_unpack_bitboards(b_packed, rank, mask_nonzero=b_nonzero)
    out = torch.matmul(x_2d.float(), bipolar.t()) * scale
    if u_proj is not None and v_proj is not None:
        out += (x_2d.float() @ v_proj.float().unsqueeze(-1)) @ u_proj.float().unsqueeze(0)
    return out.view(*orig_shape[:-1], b_packed.shape[0])


def cpu_rope_avx512(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies AVX-512 in-register RoPE rotation to input tensor x.
    """
    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_rope_avx512"):
        out = torch.zeros_like(x)
        weft_avx512.subbit_rope_avx512(x.contiguous(), cos.contiguous(), sin.contiguous(), out)
        return out

    # Fallback
    d = x.shape[-1]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def cpu_flash_attention_avx512(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Evaluates in-register tiled Flash-Attention via native AVX-512 vector registers.
    q: [batch_heads, q_len, head_dim]
    k: [batch_heads, kv_len, head_dim]
    v: [batch_heads, kv_len, head_dim]
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_flash_attention_avx512(
            q.contiguous(), k.contiguous(), v.contiguous(), out, float(sm_scale)
        )
        return out

    # Standard fallback
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def cpu_causal_flash_attention_avx512(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Evaluates fused causal Flash-Attention via native AVX-512 vector registers.
    q: [batch_heads, q_len, head_dim]
    k: [batch_heads, kv_len, head_dim]
    v: [batch_heads, kv_len, head_dim]
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_causal_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_causal_flash_attention_avx512(
            q.contiguous(), k.contiguous(), v.contiguous(), out, float(sm_scale)
        )
        return out

    # Standard fallback
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
    q_len = q.shape[-2]
    kv_len = k.shape[-2]
    mask = torch.triu(torch.full((q_len, kv_len), float('-inf'), device=q.device), diagonal=1)
    probs = torch.softmax(scores + mask, dim=-1)
    return torch.matmul(probs, v)


def cpu_paged_flash_attention_avx512(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    sm_scale: float = None,
    block_size: int = 16
) -> torch.Tensor:
    """
    Evaluates Paged KV-Cache Flash-Attention via native AVX-512 vector registers.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_paged_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_paged_flash_attention_avx512(
            q.contiguous(), k_cache.contiguous(), v_cache.contiguous(),
            block_tables.contiguous().to(torch.int32),
            context_lens.contiguous().to(torch.int32),
            out, float(sm_scale), int(block_size)
        )
        return out

    # Standard segmented fallback
    B = q.shape[0]
    out = torch.empty_like(q)
    for b in range(B):
        ctx_len = int(context_lens[b].item())
        k_parts = [k_cache[block_tables[b, i]] for i in range((ctx_len + block_size - 1) // block_size)]
        v_parts = [v_cache[block_tables[b, i]] for i in range((ctx_len + block_size - 1) // block_size)]
        k_contig = torch.cat(k_parts, dim=0)[:ctx_len]
        v_contig = torch.cat(v_parts, dim=0)[:ctx_len]
        scores = torch.matmul(q[b], k_contig.t()) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.matmul(probs, v_contig)
    return out


def cpu_int8_kv_flash_attention_avx512(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Evaluates Int8 Quantized KV-Cache Flash-Attention via native AVX-512 vector registers.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_int8_kv_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_int8_kv_flash_attention_avx512(
            q.contiguous(), k_cache.contiguous(), v_cache.contiguous(),
            k_scales.contiguous(), v_scales.contiguous(),
            out, float(sm_scale)
        )
        return out

    # Standard dequantized fallback
    k_deq = k_cache.float() * k_scales.unsqueeze(-1)
    v_deq = v_cache.float() * v_scales.unsqueeze(-1)
    scores = torch.matmul(q, k_deq.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_deq)




def cpu_int4_kv_flash_attention_avx512(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Evaluates Int4 Sub-Byte Packed KV-Cache Flash-Attention via native AVX-512 vector registers.
    k_packed and v_packed have shape (K_seq, head_dim // 2) as uint8 containing two 4-bit signed nibbles.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_int4_kv_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_int4_kv_flash_attention_avx512(
            q.contiguous(), k_packed.contiguous(), v_packed.contiguous(),
            k_scales.contiguous(), v_scales.contiguous(),
            out, float(sm_scale)
        )
        return out

    # Standard dequantized fallback for validation
    # Unpack low and high nibbles
    def unpack_nibbles(packed):
        p = packed.to(torch.int32)
        low = (p & 0x0F)
        low = torch.where(low >= 8, low - 16, low)
        high = ((p >> 4) & 0x0F)
        high = torch.where(high >= 8, high - 16, high)
        unpacked = torch.stack([low, high], dim=-1).view(*packed.shape[:-1], packed.shape[-1] * 2)
        return unpacked.float()

    k_deq = unpack_nibbles(k_packed) * k_scales.unsqueeze(-1)
    v_deq = unpack_nibbles(v_packed) * v_scales.unsqueeze(-1)
    scores = torch.matmul(q, k_deq.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_deq)


def cpu_paged_int4_kv_flash_attention_avx512(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    sm_scale: float = None,
    block_size: int = 16
) -> torch.Tensor:
    """
    Evaluates Paged Sub-Byte Int4 Quantized KV-Cache Flash-Attention via native AVX-512 vector registers.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_paged_int4_kv_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_paged_int4_kv_flash_attention_avx512(
            q.contiguous(), k_cache.contiguous(), v_cache.contiguous(),
            k_scales.contiguous(), v_scales.contiguous(),
            block_tables.contiguous().to(torch.int32),
            context_lens.contiguous().to(torch.int32),
            out, float(sm_scale), int(block_size)
        )
        return out

    def unpack_nibbles(packed):
        p = packed.to(torch.int32)
        low = (p & 0x0F)
        low = torch.where(low >= 8, low - 16, low)
        high = ((p >> 4) & 0x0F)
        high = torch.where(high >= 8, high - 16, high)
        unpacked = torch.stack([low, high], dim=-1).view(*packed.shape[:-1], packed.shape[-1] * 2)
        return unpacked.float()

    B = q.shape[0]
    out = torch.empty_like(q)
    for b in range(B):
        ctx_len = int(context_lens[b].item())
        k_parts = [k_cache[block_tables[b, i]] for i in range((ctx_len + block_size - 1) // block_size)]
        v_parts = [v_cache[block_tables[b, i]] for i in range((ctx_len + block_size - 1) // block_size)]
        ks_parts = [k_scales[block_tables[b, i]] for i in range((ctx_len + block_size - 1) // block_size)]
        vs_parts = [v_scales[block_tables[b, i]] for i in range((ctx_len + block_size - 1) // block_size)]

        k_packed_contig = torch.cat(k_parts, dim=0)[:ctx_len]
        v_packed_contig = torch.cat(v_parts, dim=0)[:ctx_len]
        ks_contig = torch.cat(ks_parts, dim=0)[:ctx_len]
        vs_contig = torch.cat(vs_parts, dim=0)[:ctx_len]

        k_deq = unpack_nibbles(k_packed_contig) * ks_contig.unsqueeze(-1)
        v_deq = unpack_nibbles(v_packed_contig) * vs_contig.unsqueeze(-1)

        scores = torch.matmul(q[b], k_deq.t()) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.matmul(probs, v_deq)
    return out


def cpu_gqa_flash_attention_avx512(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Evaluates Grouped-Query Attention (GQA) Flash-Attention via native AVX-512 vector registers.
    q: [batch, num_q_heads, q_len, head_dim]
    k: [batch, num_kv_heads, kv_len, head_dim]
    v: [batch, num_kv_heads, kv_len, head_dim]
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_gqa_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_gqa_flash_attention_avx512(
            q.contiguous(), k.contiguous(), v.contiguous(),
            out, float(sm_scale)
        )
        return out

    # Reference evaluation
    batch_size, num_q_heads, q_len, _ = q.shape
    num_kv_heads = k.shape[1]
    group_ratio = num_q_heads // num_kv_heads

    out = torch.empty_like(q)
    for b in range(batch_size):
        for qh in range(num_q_heads):
            kv_h = qh // group_ratio
            cur_q = q[b, qh]
            cur_k = k[b, kv_h]
            cur_v = v[b, kv_h]

            scores = torch.matmul(cur_q, cur_k.transpose(-1, -2)) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            out[b, qh] = torch.matmul(probs, cur_v)
    return out


def cpu_asym_int4_kv_flash_attention_avx512(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    k_zeros: torch.Tensor,
    v_zeros: torch.Tensor,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Evaluates Asymmetric Sub-Byte Int4 Quantized KV-Cache Flash-Attention via native AVX-512 vector registers.
    Activations are unsigned 4-bit integers [0, 15] packed dual-per-byte with zero-points.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_asym_int4_kv_flash_attention_avx512"):
        out = torch.empty_like(q)
        weft_avx512.subbit_asym_int4_kv_flash_attention_avx512(
            q.contiguous(), k_packed.contiguous(), v_packed.contiguous(),
            k_scales.contiguous(), v_scales.contiguous(),
            k_zeros.contiguous(), v_zeros.contiguous(),
            out, float(sm_scale)
        )
        return out

    def unpack_unsigned_nibbles(packed):
        p = packed.to(torch.int32)
        low = (p & 0x0F)
        high = ((p >> 4) & 0x0F)
        unpacked = torch.stack([low, high], dim=-1).view(*packed.shape[:-1], packed.shape[-1] * 2)
        return unpacked.float()

    k_raw = unpack_unsigned_nibbles(k_packed)
    v_raw = unpack_unsigned_nibbles(v_packed)

    k_deq = (k_raw - k_zeros.unsqueeze(-1)) * k_scales.unsqueeze(-1)
    v_deq = (v_raw - v_zeros.unsqueeze(-1)) * v_scales.unsqueeze(-1)

    scores = torch.matmul(q, k_deq.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_deq)


def cpu_fused_swiglu_gemm_avx512(
    x: torch.Tensor,
    w_gate_packed: torch.Tensor,
    w_up_packed: torch.Tensor,
    scales_gate: torch.Tensor,
    scales_up: torch.Tensor
) -> torch.Tensor:
    """
    Evaluates Fused Dual-Bitboard Gate-Up Projection GEMM with In-Register SwiGLU via native AVX-512.
    """
    batch_size = x.shape[0]
    out_features = w_gate_packed.shape[0]

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_fused_swiglu_gemm_avx512"):
        out = torch.empty(batch_size, out_features, dtype=torch.float32)
        weft_avx512.subbit_fused_swiglu_gemm_avx512(
            x.contiguous(), w_gate_packed.contiguous(), w_up_packed.contiguous(),
            scales_gate.contiguous(), scales_up.contiguous(), out
        )
        return out

    in_features = x.shape[1]
    def unpack_bitboard(packed):
        out_rows, num_words = packed.shape
        w_bin = torch.zeros(out_rows, in_features, dtype=torch.float32)
        for r in range(out_rows):
            for w in range(num_words):
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

    swish = gate * torch.sigmoid(gate)
    return swish * up


def cpu_sliding_window_flash_attention_avx512(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Sliding-Window Flash-Attention vectorized forward pass.
    Restricts attention queries qi to keys/values in [max(0, qi - W + 1), min(qi + 1, kv_len)].
    q: [batch_heads, q_len, head_dim] float32
    k: [batch_heads, kv_len, head_dim] float32
    v: [batch_heads, kv_len, head_dim] float32
    """
    batch_heads, q_len, head_dim = q.shape
    kv_len = k.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_sliding_window_flash_attention_avx512"):
        out = torch.empty(batch_heads, q_len, head_dim, dtype=torch.float32)
        weft_avx512.subbit_sliding_window_flash_attention_avx512(
            q.contiguous(), k.contiguous(), v.contiguous(), out, sm_scale, window_size
        )
        return out

    # Reference implementation
    out = torch.zeros_like(q)
    for bh in range(batch_heads):
        for qi in range(q_len):
            k_start = max(0, qi - window_size + 1)
            k_end = min(qi + 1, kv_len)
            if k_end <= k_start:
                continue
            q_vec = q[bh, qi:qi+1, :]
            k_win = k[bh, k_start:k_end, :]
            v_win = v[bh, k_start:k_end, :]
            scores = torch.matmul(q_vec, k_win.transpose(-1, -2)) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            out[bh, qi, :] = torch.matmul(probs, v_win).squeeze(0)

    return out


def cpu_fused_ternary_flash_attention_avx512(
    x: torch.Tensor,
    w_q_nz: torch.Tensor,
    w_q_sign: torch.Tensor,
    w_k_nz: torch.Tensor,
    w_k_sign: torch.Tensor,
    w_v_nz: torch.Tensor,
    w_v_sign: torch.Tensor,
    scales_q: torch.Tensor,
    scales_k: torch.Tensor,
    scales_v: torch.Tensor,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Fused Sub-Bit Ternary Flash-Attention Projection Forward.
    Projects inputs x into Q, K, V via 1.58-bit ternary bitboards and computes causal attention.
    """
    batch_heads, seq_len, in_features = x.shape
    head_dim = w_q_nz.shape[0]

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    if _AVX512_AVAILABLE and hasattr(weft_avx512, "subbit_ternary_flash_attention_avx512"):
        out = torch.empty(batch_heads, seq_len, head_dim, dtype=torch.float32)
        weft_avx512.subbit_ternary_flash_attention_avx512(
            x.contiguous(), w_q_nz.contiguous(), w_q_sign.contiguous(),
            w_k_nz.contiguous(), w_k_sign.contiguous(),
            w_v_nz.contiguous(), w_v_sign.contiguous(),
            scales_q.contiguous(), scales_k.contiguous(), scales_v.contiguous(),
            out, sm_scale
        )
        return out

    # Reference evaluation
    in_words = in_features // 64
    def project_ternary(x_in, nz_p, sign_p, scales):
        out_proj = torch.zeros(batch_heads, seq_len, head_dim, dtype=torch.float32)
        for bh in range(batch_heads):
            for t in range(seq_len):
                cur_x = torch.where(x_in[bh, t] > 0.0, 1.0, -1.0)
                for d in range(head_dim):
                    dot = 0.0
                    for w in range(in_words):
                        nz = int(nz_p[d, w].item())
                        sg = int(sign_p[d, w].item())
                        for b in range(64):
                            if (nz >> b) & 1:
                                weight_val = 1.0 if (sg >> b) & 1 else -1.0
                                dot += cur_x[w * 64 + b].item() * weight_val
                    out_proj[bh, t, d] = dot * scales[d].item()
        return out_proj

    q = project_ternary(x, w_q_nz, w_q_sign, scales_q)
    k = project_ternary(x, w_k_nz, w_k_sign, scales_k)
    v = project_ternary(x, w_v_nz, w_v_sign, scales_v)

    out = torch.zeros(batch_heads, seq_len, head_dim, dtype=torch.float32)
    for bh in range(batch_heads):
        for qi in range(seq_len):
            q_vec = q[bh, qi:qi+1, :]
            k_win = k[bh, :qi+1, :]
            v_win = v[bh, :qi+1, :]
            scores = torch.matmul(q_vec, k_win.transpose(-1, -2)) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            out[bh, qi, :] = torch.matmul(probs, v_win).squeeze(0)

    return out







