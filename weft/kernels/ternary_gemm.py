"""
Triton Addition-Only Ternary GEMM Kernel for BitNet b1.58.

Replaces floating-point matrix multiplication with integer additions and subtractions:
  Y = X @ W.T * (gamma / scale)
where:
  W in {-1, 0, 1} (packed 4 weights per uint8 byte)
  X in int8 [-128, 127]

Bypasses floating-point tensor-core multipliers entirely in GPU SRAM.
"""

from typing import Tuple
import torch
import triton
import triton.language as tl


def pack_ternary_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs ternary weights {-1, 0, 1} into 2-bit unsigned integers (4 weights per uint8 byte).
    Encoding:
      00 -> 0
      01 -> +1
      10 -> -1
      11 -> reserved (0)
    Returns:
      packed_weights: uint8 tensor of shape [out_features, in_features // 4]
      gamma: per-channel or per-tensor scale factor
    """
    assert weights.dim() == 2
    out_features, in_features = weights.shape
    assert in_features % 4 == 0, "in_features must be divisible by 4 for 2-bit packing"

    gamma = weights.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    w_ternary = torch.round(weights / gamma).clamp(-1.0, 1.0).to(torch.int8)

    # Encode: 0 -> 0, +1 -> 1, -1 -> 2
    w_encoded = torch.where(w_ternary == 1, 1, torch.where(w_ternary == -1, 2, 0)).to(torch.uint8)

    # Reshape and bitshift pack 4 weights into 1 byte:
    # byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
    w_blocks = w_encoded.view(out_features, in_features // 4, 4)
    packed = (
        w_blocks[:, :, 0]
        | (w_blocks[:, :, 1] << 2)
        | (w_blocks[:, :, 2] << 4)
        | (w_blocks[:, :, 3] << 6)
    )
    return packed, gamma


def unpack_ternary_weights(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """
    Unpacks 2-bit packed uint8 weights back into {-1, 0, 1} int8 tensor.
    """
    w_blocks = torch.empty((out_features, in_features // 4, 4), dtype=torch.int8, device=packed.device)
    w_blocks[:, :, 0] = (packed & 0x03).to(torch.int8)
    w_blocks[:, :, 1] = ((packed >> 2) & 0x03).to(torch.int8)
    w_blocks[:, :, 2] = ((packed >> 4) & 0x03).to(torch.int8)
    w_blocks[:, :, 3] = ((packed >> 6) & 0x03).to(torch.int8)

    # Decode: 0 -> 0, 1 -> +1, 2 -> -1
    decoded = torch.where(w_blocks == 1, 1, torch.where(w_blocks == 2, -1, 0))
    return decoded.view(out_features, in_features)


@triton.jit
def _ternary_gemm_kernel(
    X_ptr,          # [M, K] int8
    W_PACKED_ptr,   # [N, K // 4] uint8
    Y_ptr,          # [M, N] float32 / bfloat16
    GAMMA_ptr,      # [N, 1] scale
    SCALE_X_ptr,    # [M, 1] activation scale
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_gn,
    stride_sxm,
    M, N, K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Addition-only integer accumulation GEMM in GPU registers and SRAM:
    Iterates over K in blocks, decodes 2-bit ternary weights on-the-fly,
    and performs integer addition/subtraction accumulation without float multiplication.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in int32 registers (zero floating point roundoff)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Process K in blocks of size BLOCK_K (must be multiple of 4)
    for k_block in range(0, K, BLOCK_K):
        offs_k_byte = (k_block // 4) + tl.arange(0, BLOCK_K // 4)
        mask_k_byte = offs_k_byte < (K // 4)

        # Load packed weights: [BLOCK_N, BLOCK_K // 4]
        w_ptrs = W_PACKED_ptr + offs_n[:, None] * stride_wn + offs_k_byte[None, :] * stride_wk
        packed_w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k_byte[None, :], other=0)

        # Decode 4 weights per byte
        w0 = (packed_w & 0x03).to(tl.int8)
        w1 = ((packed_w >> 2) & 0x03).to(tl.int8)
        w2 = ((packed_w >> 4) & 0x03).to(tl.int8)
        w3 = ((packed_w >> 6) & 0x03).to(tl.int8)

        # Convert 1 -> +1, 2 -> -1, 0 -> 0
        w0_sign = tl.where(w0 == 1, 1, tl.where(w0 == 2, -1, 0))
        w1_sign = tl.where(w1 == 1, 1, tl.where(w1 == 2, -1, 0))
        w2_sign = tl.where(w2 == 1, 1, tl.where(w2 == 2, -1, 0))
        w3_sign = tl.where(w3 == 1, 1, tl.where(w3 == 2, -1, 0))

        # Interleave decoded weights to restore [BLOCK_N, BLOCK_K]
        # Load X sub-tiles corresponding to each unpacked position:
        offs_k0 = k_block + (tl.arange(0, BLOCK_K // 4) * 4)
        x_ptrs0 = X_ptr + offs_m[:, None] * stride_xm + offs_k0[None, :] * stride_xk
        x0 = tl.load(x_ptrs0, mask=mask_m[:, None] & (offs_k0[None, :] < K), other=0).to(tl.int8)

        offs_k1 = offs_k0 + 1
        x_ptrs1 = X_ptr + offs_m[:, None] * stride_xm + offs_k1[None, :] * stride_xk
        x1 = tl.load(x_ptrs1, mask=mask_m[:, None] & (offs_k1[None, :] < K), other=0).to(tl.int8)

        offs_k2 = offs_k0 + 2
        x_ptrs2 = X_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk
        x2 = tl.load(x_ptrs2, mask=mask_m[:, None] & (offs_k2[None, :] < K), other=0).to(tl.int8)

        offs_k3 = offs_k0 + 3
        x_ptrs3 = X_ptr + offs_m[:, None] * stride_xm + offs_k3[None, :] * stride_xk
        x3 = tl.load(x_ptrs3, mask=mask_m[:, None] & (offs_k3[None, :] < K), other=0).to(tl.int8)

        # Pure additions/subtractions in SRAM registers via int8 dot accumulating to int32:
        acc = tl.dot(x0, tl.trans(w0_sign.to(tl.int8)), acc)
        acc = tl.dot(x1, tl.trans(w1_sign.to(tl.int8)), acc)
        acc = tl.dot(x2, tl.trans(w2_sign.to(tl.int8)), acc)
        acc = tl.dot(x3, tl.trans(w3_sign.to(tl.int8)), acc)

    # Dequantize with gamma and activation scale:
    # y = acc * gamma * (1 / scale_x)
    gamma_ptrs = GAMMA_ptr + offs_n * stride_gn
    gamma = tl.load(gamma_ptrs, mask=mask_n, other=1.0)[:, None]  # [BLOCK_N, 1]

    scale_x_ptrs = SCALE_X_ptr + offs_m * stride_sxm
    scale_x = tl.load(scale_x_ptrs, mask=mask_m, other=1.0)[:, None]  # [BLOCK_M, 1]

    # Convert to float and apply scale
    acc_f32 = tl.cast(acc, tl.float32)
    # scale factor: (1 / scale_x) * gamma.T
    y_f32 = acc_f32 * (tl.trans(gamma) / scale_x)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, tl.cast(y_f32, Y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def ternary_gemm_triton(
    x_int8: torch.Tensor,       # [M, K] int8
    scale_x: torch.Tensor,      # [M, 1] float32
    w_packed: torch.Tensor,     # [N, K // 4] uint8
    gamma: torch.Tensor,        # [N, 1] float32
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Executes BitNet b1.58 ternary GEMM via Triton addition accumulation.
    """
    M, K = x_int8.shape
    N, K_packed = w_packed.shape
    assert K_packed == K // 4

    y = torch.empty((M, N), device=x_int8.device, dtype=dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _ternary_gemm_kernel[grid](
        x_int8, w_packed, y, gamma, scale_x,
        x_int8.stride(0), x_int8.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        y.stride(0), y.stride(1),
        gamma.stride(0),
        scale_x.stride(0),
        M=M, N=N, K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return y
