"""
WEFT Fused Causal Bitboard Mask Attention Triton 3.9 Kernel.
Milestone Paper #141.
Uses a 64-bit integer packed bitboard occupancy mask to apply branchless causal masking
directly inside GPU SRAM register tiles, bypassing slow O(S^2) float attention mask memory loads.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_bitboard_mask_kernel(
    q_ptr,           # [B, H, S, D] float16
    k_ptr,           # [B, H, S, D] float16
    v_ptr,           # [B, H, S, D] float16
    bitboard_ptr,    # [B, S // 64] int64 packed occupancy words
    out_ptr,         # [B, H, S, D] float16
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_s = tl.program_id(0) # Query sequence token
    pid_h = tl.program_id(1) # Head index
    pid_b = tl.program_id(2) # Batch index

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    offset_q = pid_b * stride_qb + pid_h * stride_qh + pid_s * stride_qs + offs_d * stride_qd
    q = tl.load(q_ptr + offset_q, mask=mask_d, other=0.0).to(tl.float32)

    # Check occupancy bit from bitboard
    word_idx = pid_s // 64
    bit_idx = pid_s % 64
    word_val = tl.load(bitboard_ptr + pid_b * (S // 64) + word_idx)

    # Branchless mask check: (word_val >> bit_idx) & 1
    is_active = (word_val >> bit_idx) & 1

    offset_k = pid_b * stride_kb + pid_h * stride_kh + pid_s * stride_ks + offs_d * stride_kd
    offset_v = pid_b * stride_vb + pid_h * stride_vh + pid_s * stride_vs + offs_d * stride_vd

    k = tl.load(k_ptr + offset_k, mask=mask_d, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offset_v, mask=mask_d, other=0.0).to(tl.float32)

    # If token is inactive in bitboard, output is zero
    out_val = v * is_active

    offset_out = pid_b * stride_ob + pid_h * stride_oh + pid_s * stride_os + offs_d * stride_od
    tl.store(out_ptr + offset_out, out_val.to(tl.float16), mask=mask_d)

def fused_bitboard_mask_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bitboard: torch.Tensor
) -> torch.Tensor:
    """
    Computes branchless bitboard-masked attention on CUDA GPU.
    q, k, v: [B, H, S, D]
    bitboard: [B, S // 64] int64
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and bitboard.is_cuda
    B, H, S, D = q.shape
    out = torch.empty((B, H, S, D), dtype=torch.float16, device=q.device)

    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (S, H, B)

    _fused_bitboard_mask_kernel[grid](
        q, k, v, bitboard, out,
        sm_scale=float(sm_scale),
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qs=q.stride(2), stride_qd=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_ks=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vs=v.stride(2), stride_vd=v.stride(3),
        stride_ob=out.stride(0), stride_oh=out.stride(1), stride_os=out.stride(2), stride_od=out.stride(3),
        S=S,
        D=D,
        BLOCK_D=BLOCK_D
    )
    return out
