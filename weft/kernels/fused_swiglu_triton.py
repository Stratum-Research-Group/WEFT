"""
WEFT Fused SwiGLU / GeGLU Activation Triton 3.9 Kernel.
Milestone Paper #134.
Executes fused gating and activation: out = (gate * sigmoid(gate)) * up
directly inside SRAM register blocks, avoiding high-bandwidth memory (HBM) round-trips.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_swiglu_kernel(
    gate_ptr,    # [M, N]
    up_ptr,      # [M, N]
    out_ptr,     # [M, N]
    stride_m, stride_n,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N

    offset = pid_m * stride_m + offs_n * stride_n

    # Load gate and up projection tokens
    gate = tl.load(gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    # Swish / SiLU activation: x * sigmoid(x)
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    # Elementwise multiplication with up-projection
    res = silu_gate * up

    # Store back to output in float16
    tl.store(out_ptr + offset, res.to(tl.float16), mask=mask)

def fused_swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Computes fused SwiGLU: (gate * sigmoid(gate)) * up on CUDA GPU.
    """
    assert gate.is_cuda and up.is_cuda, "Inputs must be on CUDA GPU"
    assert gate.shape == up.shape, "Gate and up tensors must have matching shapes"
    
    orig_shape = gate.shape
    M = gate.numel() // orig_shape[-1]
    N = orig_shape[-1]

    gate_2d = gate.view(M, N)
    up_2d = up.view(M, N)
    out = torch.empty_like(gate_2d, dtype=torch.float16)

    BLOCK_SIZE = 1024
    grid = (M, triton.cdiv(N, BLOCK_SIZE))

    _fused_swiglu_kernel[grid](
        gate_2d,
        up_2d,
        out,
        gate_2d.stride(0), gate_2d.stride(1),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out.view(orig_shape)
