"""
WEFT Fused StripedHyena Long-Conv Filter Triton 3.9 Kernel.
Milestone Paper #138.
Computes 1D causal long-range convolution: y = u * h
directly in GPU SRAM blocks for sequence contexts up to 131,072 base pairs (131K bp).
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_hyena_conv_kernel(
    u_ptr,           # Input sequence [B, L]
    h_ptr,           # Filter kernel [K]
    out_ptr,         # Output sequence [B, L]
    L: tl.constexpr, # Sequence length
    K: tl.constexpr, # Filter length (e.g. 64)
    BLOCK_L: tl.constexpr
):
    pid_b = tl.program_id(0) # Batch index
    pid_blk = tl.program_id(1) # Block index along sequence

    offs_l = pid_blk * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L

    base_u = pid_b * L
    base_out = pid_b * L

    # Direct causal sliding dot product per element
    for idx in range(0, BLOCK_L):
        curr_pos = pid_blk * BLOCK_L + idx
        if curr_pos < L:
            acc = 0.0
            for k in range(0, K):
                src_pos = curr_pos - k
                if src_pos >= 0:
                    u_val = tl.load(u_ptr + base_u + src_pos).to(tl.float32)
                    h_val = tl.load(h_ptr + k).to(tl.float32)
                    acc += u_val * h_val
            tl.store(out_ptr + base_out + curr_pos, acc.to(tl.float16))

def hyena_long_conv_triton(
    u: torch.Tensor,
    h: torch.Tensor
) -> torch.Tensor:
    """
    Computes 1D causal long-convolution on CUDA GPU.
    u: [B, L] input sequence (float16)
    h: [K] filter weights (float16)
    """
    assert u.is_cuda and h.is_cuda, "Inputs must be on CUDA GPU"
    assert u.ndim == 2, "Input u must be [B, L]"
    assert h.ndim == 1, "Filter h must be [K]"

    B, L = u.shape
    K = h.shape[0]
    out = torch.empty((B, L), dtype=torch.float16, device=u.device)

    BLOCK_L = 128
    grid = (B, triton.cdiv(L, BLOCK_L))

    _fused_hyena_conv_kernel[grid](
        u.contiguous(),
        h.contiguous(),
        out,
        L=L,
        K=K,
        BLOCK_L=BLOCK_L
    )
    return out
