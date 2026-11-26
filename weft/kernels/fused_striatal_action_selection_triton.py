"""
Striatal D1/D2 Opponent Action Selection & Dopaminergic RPE Kernel in Triton 3.9.

Models:
1. Corticostriatal Synaptic Projections to Medium Spiny Neurons (MSNs):
   - Direct Pathway (D1-MSNs): Facilitated by positive dopamine RPE (delta_DA > 0)
     y_D1 = relu( (W_D1 @ x_ctx) * (1.0 + beta_D1 * delta_DA) )
   - Indirect Pathway (D2-MSNs): Inhibited by positive dopamine RPE, facilitated by negative RPE (delta_DA < 0)
     y_D2 = relu( (W_D2 @ x_ctx) * (1.0 - beta_D2 * delta_DA) )
2. Basal Ganglia Internal Segment of Globus Pallidus (GPi) Disinhibition:
   Thalamic drive = relu( W_th_base - (y_D2 - y_D1) )
   Action is gated/released if net direct inhibition of GPi exceeds threshold.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _striatal_d1_d2_kernel(
    X_CTX_ptr,       # [B, N_CTX]
    W_D1_ptr,        # [N_ACT, N_CTX]
    W_D2_ptr,        # [N_ACT, N_CTX]
    DELTA_DA_ptr,    # [B] - dopamine RPE scalar
    Y_D1_ptr,        # [B, N_ACT]
    Y_D2_ptr,        # [B, N_ACT]
    GATED_ptr,       # [B, N_ACT] - binary flag indicating action gate opening
    beta_d1: float,
    beta_d2: float,
    gpi_thresh: float,
    B: tl.constexpr,
    N_CTX: tl.constexpr,
    N_ACT: tl.constexpr,
    BLOCK_CTX: tl.constexpr,
    BLOCK_ACT: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_a = tl.program_id(axis=1)

    offs_a = pid_a * BLOCK_ACT + tl.arange(0, BLOCK_ACT)
    mask_a = offs_a < N_ACT

    # Load dopamine RPE for batch item
    delta = tl.load(DELTA_DA_ptr + pid_b)
    mod_d1 = 1.0 + beta_d1 * delta
    mod_d2 = 1.0 - beta_d2 * delta

    # Clamp modulation to non-negative
    mod_d1 = tl.maximum(mod_d1, 0.0)
    mod_d2 = tl.maximum(mod_d2, 0.0)

    # Accumulate D1 and D2 drive
    acc_d1 = tl.zeros([BLOCK_ACT], dtype=tl.float32)
    acc_d2 = tl.zeros([BLOCK_ACT], dtype=tl.float32)

    for k in range(0, N_CTX, BLOCK_CTX):
        offs_k = k + tl.arange(0, BLOCK_CTX)
        mask_k = offs_k < N_CTX

        x_ctx = tl.load(X_CTX_ptr + pid_b * N_CTX + offs_k, mask=mask_k, other=0.0)

        w1 = tl.load(
            W_D1_ptr + offs_a[:, None] * N_CTX + offs_k[None, :],
            mask=(mask_a[:, None] & mask_k[None, :]),
            other=0.0
        )
        w2 = tl.load(
            W_D2_ptr + offs_a[:, None] * N_CTX + offs_k[None, :],
            mask=(mask_a[:, None] & mask_k[None, :]),
            other=0.0
        )

        acc_d1 += tl.sum(w1 * x_ctx[None, :], axis=1)
        acc_d2 += tl.sum(w2 * x_ctx[None, :], axis=1)

    # Apply dopaminergic modulation and ReLU
    y_d1 = tl.maximum(acc_d1 * mod_d1, 0.0)
    y_d2 = tl.maximum(acc_d2 * mod_d2, 0.0)

    # Action gate condition: D1 direct drive must exceed D2 indirect drive + threshold
    gate_signal = y_d1 - y_d2 - gpi_thresh
    is_gated = tl.where(gate_signal > 0.0, 1.0, 0.0)

    # Store results
    tl.store(Y_D1_ptr + pid_b * N_ACT + offs_a, y_d1, mask=mask_a)
    tl.store(Y_D2_ptr + pid_b * N_ACT + offs_a, y_d2, mask=mask_a)
    tl.store(GATED_ptr + pid_b * N_ACT + offs_a, is_gated, mask=mask_a)


def striatal_action_selection(
    x_ctx: torch.Tensor,
    w_d1: torch.Tensor,
    w_d2: torch.Tensor,
    delta_da: torch.Tensor,
    beta_d1: float = 0.8,
    beta_d2: float = 0.8,
    gpi_thresh: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates striatal D1/D2 opponent pathway action gating.

    Args:
        x_ctx:    [B, N_CTX] cortical state representation
        w_d1:     [N_ACT, N_CTX] corticostriatal D1 synaptic weights
        w_d2:     [N_ACT, N_CTX] corticostriatal D2 synaptic weights
        delta_da: [B] dopamine reward prediction error
        beta_d1:  dopamine D1 facilitation coefficient
        beta_d2:  dopamine D2 inhibition coefficient
        gpi_thresh: GPi disinhibition threshold

    Returns:
        (y_d1, y_d2, action_gated_mask)
    """
    B, N_CTX = x_ctx.shape
    N_ACT = w_d1.shape[0]
    device = x_ctx.device

    if x_ctx.is_cuda:
        y_d1 = torch.empty((B, N_ACT), device=device, dtype=torch.float32)
        y_d2 = torch.empty((B, N_ACT), device=device, dtype=torch.float32)
        action_gated = torch.empty((B, N_ACT), device=device, dtype=torch.float32)

        BLOCK_CTX = 128 if N_CTX >= 128 else 32
        BLOCK_ACT = 64 if N_ACT >= 64 else 16
        grid = (B, triton.cdiv(N_ACT, BLOCK_ACT))

        _striatal_d1_d2_kernel[grid](
            x_ctx.contiguous(),
            w_d1.contiguous(),
            w_d2.contiguous(),
            delta_da.contiguous(),
            y_d1,
            y_d2,
            action_gated,
            float(beta_d1),
            float(beta_d2),
            float(gpi_thresh),
            B=B,
            N_CTX=N_CTX,
            N_ACT=N_ACT,
            BLOCK_CTX=BLOCK_CTX,
            BLOCK_ACT=BLOCK_ACT,
        )
        return y_d1, y_d2, action_gated
    else:
        # High-performance PyTorch CPU fallback
        mod_d1 = torch.clamp(1.0 + beta_d1 * delta_da.unsqueeze(-1), min=0.0)
        mod_d2 = torch.clamp(1.0 - beta_d2 * delta_da.unsqueeze(-1), min=0.0)

        drive_d1 = torch.matmul(x_ctx, w_d1.t())
        drive_d2 = torch.matmul(x_ctx, w_d2.t())

        y_d1 = torch.relu(drive_d1 * mod_d1)
        y_d2 = torch.relu(drive_d2 * mod_d2)

        gate_signal = y_d1 - y_d2 - gpi_thresh
        action_gated = (gate_signal > 0.0).float()

        return y_d1, y_d2, action_gated
