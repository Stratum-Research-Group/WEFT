# SRAM Bank Padding (+8) for 32-bank conflict-free shared memory access
BANK_PADDING: int = 8
"""
Triton SRAM-Fused INT8 Dequantizing GEMM / GEMV Kernel.

Performs fused dequantization and matrix-vector / matrix-matrix multiplication:
    y = x @ (w_int8.to(bfloat16) * scale) + bias

Weights are loaded as INT8 directly from DRAM into SRAM/registers, cutting
memory bandwidth in half compared to full bfloat16 weight reads. Dequantization
happens in-register before the MMA accumulation.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_int8_gemm_kernel(
    x_ptr,
    w_int8_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    x: [M, K] in bfloat16
    w_int8: [N, K] in int8 (transposed layout: out_features x in_features)
    scale: [N, 1] in bfloat16
    bias: [N] in bfloat16 or None
    out: [M, N] in bfloat16
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Output accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Scales for this block of N outputs
    mask_n = offs_n < N
    scales = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)

    # Loop over K tiles
    for k in range(0, K, BLOCK_K):
        current_k = k + offs_k
        mask_k = current_k < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk)
        x_mask = (offs_m[:, None] < M) & (mask_k[None, :])
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # Load W_int8 tile: [BLOCK_N, BLOCK_K]
        w_ptrs = w_int8_ptr + (offs_n[:, None] * stride_wn + current_k[None, :] * stride_wk)
        w_mask = (mask_n[:, None]) & (mask_k[None, :])
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.float32)

        # In-register dequantization
        # w_dequant: [BLOCK_N, BLOCK_K]
        w_dequant = w_int8 * scales[:, None]

        # MMA: acc += x @ w_dequant.T
        # x is [BLOCK_M, BLOCK_K], w_dequant.T is [BLOCK_K, BLOCK_N]
        acc += tl.dot(x, tl.trans(w_dequant))

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Store result back in bfloat16
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def fused_int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused INT8 Linear layer execution using Triton SRAM dequantization.

    Args:
        x: Input activations of shape [..., in_features] in torch.bfloat16
        weight_int8: INT8 weight of shape [out_features, in_features]
        scale: per-channel scale of shape [out_features, 1] or [out_features]
        bias: optional bias of shape [out_features]

    Returns:
        Output tensor of shape [..., out_features] in torch.bfloat16
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, K = x_2d.shape
    N = weight_int8.shape[0]

    assert weight_int8.shape[1] == K, f"Shape mismatch: x K={K}, w K={weight_int8.shape[1]}"
    assert scale.numel() == N, f"Scale numel mismatch: expected {N}, got {scale.numel()}"

    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else x

    _fused_int8_gemm_kernel[grid](
        x_2d,
        weight_int8,
        scale.squeeze(),
        bias_ptr,
        out,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        weight_int8.stride(0),
        weight_int8.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=has_bias,
    )

    return out.view(*orig_shape[:-1], N)
