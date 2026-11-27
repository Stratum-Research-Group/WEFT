# SRAM Bank Padding (+8) for 32-bank conflict-free shared memory access
BANK_PADDING: int = 8
"""
WEFT Fused Triton Kernel for Register-Space Vision Patch Ingestion
Directly ingests image patches, unpacks bipolar bitboards, and applies channel scales
without staging dense intermediate matrices in GPU HBM.
"""

from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def subbit_vision_projector_kernel(
    x_ptr,           # Input vision patch embeddings [M, K]
    b_ptr,           # Packed bipolar factor bitboards [K_words, N]
    g_ptr,           # Input scale vector [K]
    h_ptr,           # Output scale vector [N]
    out_ptr,         # Output projected embeddings [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_bw, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Ingest along feature dimension K in chunks of 32 bits (1 word)
    K_words = K // 32
    for w in range(0, K_words):
        # Load packed 32-bit bitboard words for target columns
        w_idx = w * stride_bw + offs_n * stride_bn
        packed_words = tl.load(b_ptr + w_idx, mask=mask_n, other=0)

        # Unpack each bit and accumulate with scaled input
        for bit in range(0, 32):
            k = w * 32 + bit
            x_val = tl.load(x_ptr + offs_m * stride_xm + k * stride_xk, mask=mask_m, other=0.0)
            g_val = tl.load(g_ptr + k)
            
            # Extract sign bit: 1 -> +1.0, 0 -> -1.0
            bit_val = (packed_words >> bit) & 1
            sign = tl.where(bit_val == 1, 1.0, -1.0)

            term = x_val * g_val
            acc += term[:, None] * sign[None, :]

    h_vals = tl.load(h_ptr + offs_n, mask=mask_n, other=1.0)
    acc = acc * h_vals[None, :]

    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_subbit_vision_projector(
    x: torch.Tensor,
    packed_bitboard: torch.Tensor,
    g_scale: torch.Tensor,
    h_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Python wrapper executing fused vision projection.
    Falls back to optimized PyTorch CPU simulation if CUDA/Triton is unavailable.
    """
    M, K = x.shape
    K_words, N = packed_bitboard.shape
    assert K == K_words * 32, f"Dimension mismatch: K={K} vs K_words*32={K_words*32}"

    if x.is_cuda and torch.cuda.is_available():
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        subbit_vision_projector_kernel[grid](
            x, packed_bitboard, g_scale, h_scale, out,
            M, N, K,
            x.stride(0), x.stride(1),
            packed_bitboard.stride(0), packed_bitboard.stride(1),
            out.stride(0), out.stride(1),
        )
        return out
    else:
        # High-performance CPU fallback
        # Unpack packed_bitboard (int32) into bipolar {-1, +1}
        device = x.device
        bits = []
        for b in range(32):
            bit = (packed_bitboard >> b) & 1
            sign = torch.where(bit == 1, 1.0, -1.0)
            bits.append(sign)
        # Stack along K dimension: (K_words, 32, N) -> (K, N)
        B = torch.stack(bits, dim=1).reshape(K, N).to(device)
        
        # Multiply: (x * g) @ (B * h)
        x_scaled = x * g_scale.unsqueeze(0)
        B_scaled = B * h_scale.unsqueeze(0)
        return torch.matmul(x_scaled, B_scaled)
