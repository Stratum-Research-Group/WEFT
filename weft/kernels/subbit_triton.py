"""
High-performance fused Triton kernel for Sub-Bit Binary MatMul.
Executes bipolar GEMM on packed uint32 bitboards directly on GPU:
y = x @ B_unpacked^T
where B_unpacked in {-1.0, +1.0}^(N x K) is stored as (N, K // 32) uint32 words.
Bipolar decode: bit 1 -> +1.0, bit 0 -> -1.0.
Computes in registers without dense materialization in HBM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _subbit_gemm_kernel(
    x_ptr,           # (M, K) float16 / bfloat16 / float32
    b_packed_ptr,    # (N, K_words) uint32
    out_ptr,         # (M, N) float32
    M, N, K,
    stride_xm, stride_xk,
    stride_bn, stride_bkw,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_words = K // 32

    # Loop over 32-bit words along the K dimension
    for kw in range(K_words):
        # Load packed bitboard slice for BLOCK_N: (BLOCK_N,) uint32
        b_ptrs = b_packed_ptr + (offs_n * stride_bn + kw * stride_bkw)
        b_bits = tl.load(b_ptrs, mask=mask_n, other=0)

        # Unpack each of the 32 bits and accumulate inner product with x
        for bit_idx in range(32):
            k_curr = kw * 32 + bit_idx
            # Load x slice for BLOCK_M: (BLOCK_M,)
            x_ptrs = x_ptr + (offs_m * stride_xm + k_curr * stride_xk)
            x_val = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            # Extract bit: 1 -> +1.0, 0 -> -1.0
            bit = (b_bits >> bit_idx) & 1
            # bipolar = 2 * bit - 1
            sign = tl.where(bit == 1, 1.0, -1.0)

            # Outer product accumulation: (BLOCK_M, 1) * (1, BLOCK_N)
            acc += x_val[:, None] * sign[None, :]

    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def subbit_matmul(x: torch.Tensor, b_packed: torch.Tensor, K: int) -> torch.Tensor:
    """
    Computes y = x @ B_unpacked^T
    Args:
        x: (M, K) input tensor
        b_packed: (N, K // 32) uint32 bitboard tensor
        K: dimension along reduction axis (must be multiple of 32)
    Returns:
        (M, N) float32 result tensor
    """
    assert x.is_cuda and b_packed.is_cuda, "Tensors must be on CUDA"
    assert x.shape[-1] == K, f"Expected x.shape[-1] == {K}, got {x.shape[-1]}"
    
    orig_shape = x.shape
    M = x.numel() // K
    N = b_packed.shape[0]

    x_2d = x.reshape(M, K).contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _subbit_gemm_kernel[grid](
        x_2d,
        b_packed,
        out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out.view(*orig_shape[:-1], N)
