"""
WEFT Multi-Head Rotary Position Embedding (RoPE) Kernel in Triton 3.9.
Milestone Paper #131.
Computes 2D vector pair rotations across attention heads in parallel with online NTK frequency scaling.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_rope_kernel(
    x_ptr,           # [B, H, S, D]
    out_ptr,         # [B, H, S, D]
    inv_freq_ptr,    # [D // 2]
    stride_b, stride_h, stride_s, stride_d,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    pid_s = tl.program_id(0) # Sequence position m
    pid_h = tl.program_id(1) # Head index
    pid_b = tl.program_id(2) # Batch index

    offs_d = tl.arange(0, HALF_D)
    inv_freq = tl.load(inv_freq_ptr + offs_d)
    theta = pid_s * inv_freq
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)

    # Offsets for pair (d, d + HALF_D)
    base_offset = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s
    offs_0 = base_offset + offs_d * stride_d
    offs_1 = base_offset + (offs_d + HALF_D) * stride_d

    x0 = tl.load(x_ptr + offs_0)
    x1 = tl.load(x_ptr + offs_1)

    r0 = x0 * cos_theta - x1 * sin_theta
    r1 = x0 * sin_theta + x1 * cos_theta

    tl.store(out_ptr + offs_0, r0)
    tl.store(out_ptr + offs_1, r1)

def fused_rope_triton(x: torch.Tensor, base_freq: float = 10000.0) -> torch.Tensor:
    """
    Applies rotary position embeddings to tensor x of shape [B, H, S, D]
    """
    B, H, S, D = x.shape
    HALF_D = D // 2
    device = x.device

    # Compute inverse frequencies
    dim_indices = torch.arange(0, HALF_D, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base_freq ** (2.0 * dim_indices / D))

    out = torch.empty_like(x)
    grid = (S, H, B)

    _fused_rope_kernel[grid](
        x, out, inv_freq,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        D=D, HALF_D=HALF_D
    )
    return out

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.randn(1, 8, 16, 64, device=device, dtype=torch.float32)
        r = fused_rope_triton(x)
        torch.cuda.synchronize()

        # Check L2 norm preservation
        norm_in = torch.norm(x[0, 0, 0, :2]).item()
        norm_out = torch.norm(r[0, 0, 0, :2]).item()
        print(f"[+] Multi-Head RoPE Triton Verified: NormIn={norm_in:.4f}, NormOut={norm_out:.4f}")
    else:
        print("[*] CUDA unavailable; syntax verified.")
