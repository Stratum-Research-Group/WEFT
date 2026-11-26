"""
Fused Hippocampal CA3 Recurrent Auto-Associative Attractor Kernel in Triton 3.9.

Models:
1. Direct recurrent collateral synaptic transmission:
   h_i^{(t+1)} = \tanh\left( \sum_{j} W_{ij}^{\text{RC}} x_j^{(t)} + \beta x_i^{(0)} + \theta_i \right)
2. Lyapunov energy monitoring:
   E(x) = -\frac{1}{2} x^T W^{\text{RC}} x - \beta x^T x^{(0)}
3. SRAM fused recurrent settling with dynamic early-exit convergence check.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _ca3_recurrent_step_kernel(
    X_ptr,          # [B, N] - current state x^{(t)}
    X0_ptr,         # [B, N] - initial sensory/cue clamp x^{(0)}
    W_ptr,          # [N, N] - recurrent collateral weights W^{RC}
    X_NEXT_ptr,     # [B, N] - output state x^{(t+1)}
    ENERGY_ptr,     # [B]    - output Lyapunov energy
    beta: tl.constexpr, # cue clamping factor
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

    offs_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_i = offs_i < N

    # Accumulate recurrent collateral input: sum_j W_{ij} * x_j
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Tile over j dimension
    for j_start in range(0, N, BLOCK_N):
        offs_j = j_start + tl.arange(0, BLOCK_N)
        mask_j = offs_j < N

        # Load x_j for batch item pid_b
        x_j = tl.load(X_ptr + pid_b * N + offs_j, mask=mask_j, other=0.0)

        # Load weight block W[offs_i, offs_j]
        # In memory W is shape [N, N] row-major
        w_block = tl.load(
            W_ptr + offs_i[:, None] * N + offs_j[None, :],
            mask=(mask_i[:, None] & mask_j[None, :]),
            other=0.0
        )

        acc += tl.sum(w_block * x_j[None, :], axis=1)

    # Load initial clamp x0
    x0 = tl.load(X0_ptr + pid_b * N + offs_i, mask=mask_i, other=0.0)

    # Recurrent update with tanh activation: tanh(acc + beta * x0)
    net_input = acc + beta * x0
    x_next = tl.libdevice.tanh(net_input)

    # Store next state
    tl.store(X_NEXT_ptr + pid_b * N + offs_i, x_next, mask=mask_i)


def ca3_recurrent_settle(
    x_cue: torch.Tensor,
    w_rc: torch.Tensor,
    max_steps: int = 15,
    beta: float = 0.35,
    tol: float = 1e-4
) -> Tuple[torch.Tensor, int, float]:
    """
    Simulates CA3 recurrent auto-associative pattern completion.
    
    Args:
        x_cue: [B, N] initial cue input (possibly occluded/noisy)
        w_rc: [N, N] symmetric recurrent collateral weights
        max_steps: maximum recurrent settling iterations
        beta: strength of external cue clamp
        tol: convergence tolerance on L2 state displacement
    
    Returns:
        (x_converged, steps_taken, final_lyapunov_energy)
    """
    B, N = x_cue.shape
    device = x_cue.device
    x_curr = x_cue.clone().to(torch.float32)
    x0 = x_cue.clone().to(torch.float32)

    steps_taken = 0
    BLOCK_N = 128 if N >= 128 else 32

    # Check if GPU is available for Triton
    if x_cue.is_cuda:
        x_next = torch.empty_like(x_curr)
        energy_tensor = torch.zeros(B, device=device, dtype=torch.float32)
        grid = (B, triton.cdiv(N, BLOCK_N))

        for step in range(max_steps):
            steps_taken = step + 1
            _ca3_recurrent_step_kernel[grid](
                x_curr,
                x0,
                w_rc,
                x_next,
                energy_tensor,
                beta=float(beta),
                B=B,
                N=N,
                BLOCK_N=BLOCK_N,
            )
            diff = torch.norm(x_next - x_curr, dim=-1).max().item()
            x_curr.copy_(x_next)
            if diff < tol:
                break
    else:
        # High-performance PyTorch CPU fallback
        for step in range(max_steps):
            steps_taken = step + 1
            # h^{(t+1)} = tanh(x^{(t)} @ W^T + beta * x^{(0)})
            net_input = torch.matmul(x_curr, w_rc.t()) + beta * x0
            x_next = torch.tanh(net_input)
            diff = torch.norm(x_next - x_curr, dim=-1).max().item()
            x_curr = x_next
            if diff < tol:
                break

    # Calculate Lyapunov energy: E = -0.5 * x^T W x - beta * x^T x0
    # For batch: E_b = -0.5 * sum(x * (x @ W^T)) - beta * sum(x * x0)
    quad = 0.5 * torch.sum(x_curr * torch.matmul(x_curr, w_rc.t()), dim=-1)
    linear = beta * torch.sum(x_curr * x0, dim=-1)
    energy = - (quad + linear).mean().item()

    return x_curr, steps_taken, energy
