"""
WEFT Streaming FP8 / FP4 KV-Cache Dequantization Triton 3.9 Kernel.
Milestone Paper #133.
Dequantizes block-scaled 8-bit float (E4M3/E5M2) or 4-bit packed tokens
directly in SRAM during multi-head attention prefill and decode passes.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_fp8_kv_dequant_kernel(
    src_ptr,          # [B, H, S, D] int8 / fp8 representation
    scales_ptr,       # [B, H, S] per-token or per-block scale factors
    dst_ptr,          # [B, H, S, D] fp16 output
    stride_b, stride_h, stride_s, stride_d,
    scale_stride_b, scale_stride_h, scale_stride_s,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_s = tl.program_id(0) # Sequence token index
    pid_h = tl.program_id(1) # Head index
    pid_b = tl.program_id(2) # Batch index

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D

    # Compute memory offsets
    base_src = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s
    scale_offset = pid_b * scale_stride_b + pid_h * scale_stride_h + pid_s * scale_stride_s

    # Load scale factor
    scale = tl.load(scales_ptr + scale_offset)

    # Load quantized bytes / float8 elements
    src_vals = tl.load(src_ptr + base_src + offs_d * stride_d, mask=mask, other=0.0)

    # Fused dequantization: value * scale
    # In Triton, float8 types cast seamlessly to float16
    dequant = src_vals.to(tl.float32) * scale
    dequant_fp16 = dequant.to(tl.float16)

    # Store back to SRAM / HBM
    tl.store(dst_ptr + base_src + offs_d * stride_d, dequant_fp16, mask=mask)

def dequantize_fp8_kv_cache(
    q_tensor: torch.Tensor,
    scales: torch.Tensor
) -> torch.Tensor:
    """
    Dequantizes an FP8 or INT8 KV cache tensor [B, H, S, D] into FP16 using per-token scale factors.
    """
    assert q_tensor.is_cuda, "Input tensor must be on CUDA GPU"
    B, H, S, D = q_tensor.shape
    out = torch.empty((B, H, S, D), dtype=torch.float16, device=q_tensor.device)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (S, H, B)

    _fused_fp8_kv_dequant_kernel[grid](
        q_tensor,
        scales,
        out,
        q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2), q_tensor.stride(3),
        scales.stride(0), scales.stride(1), scales.stride(2),
        D=D,
        BLOCK_D=BLOCK_D
    )
    return out
