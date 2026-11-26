"""
Fused Cortico-Ponto-Cerebellar Forward Model & Sensory Reafference Predictor Kernel in Triton 3.9.

Models:
1. Motor Cortex (M1) Efference Copy Branching:
   - Primary motor command descends down corticospinal tract: u_motor [B, D_M1]
   - Collateral copies project to Pontine Nuclei (PN): u_pontine = W_ponto @ u_motor
2. Pontine to Cerebellar Granule Mossy Fiber Expansion:
   - x_mossy = relu(W_pn_to_mf @ u_pontine)
3. Cerebellar Internal Model State Prediction:
   - Purkinje cell population integrates parallel fiber activity to predict expected
     sensory reafference:
     y_hat_sensory = W_dcn @ (1.0 - sigmoid(W_pc @ x_mossy))
4. Sensory Reafference Cancellation (Smith Predictor):
   - e_reafference = y_actual - y_hat_sensory
   - Exafference (unexpected external sensory events) passes through to conscious cortex,
     while self-generated sensations are precisely cancelled.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_forward_sensory_predictor(
    u_motor: torch.Tensor,
    y_actual: torch.Tensor,
    w_forward: torch.Tensor,
    cancellation_gain: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Simulates cerebro-cerebellar forward model prediction and sensory reafference cancellation.

    Args:
        u_motor: [B, D_MOTOR] motor efference copy
        y_actual: [B, D_SENSORY] actual incoming sensory feedback (delayed)
        w_forward: [D_SENSORY, D_MOTOR] learned forward internal model matrix
        cancellation_gain: scalar gain for cancellation subtraction

    Returns:
        (y_pred_reafference, e_sensory_discrepancy, mean_cancellation_ratio)
    """
    # Linear-nonlinear forward internal model prediction
    # y_pred = u_motor @ W_forward^T
    y_pred = torch.matmul(u_motor, w_forward.t())

    # Sensory cancellation: e = y_actual - gain * y_pred
    e_discrepancy = y_actual - cancellation_gain * y_pred

    # Cancellation ratio: 1.0 - (norm(e) / (norm(y_actual) + eps))
    norm_actual = torch.norm(y_actual, dim=-1)
    norm_error = torch.norm(e_discrepancy, dim=-1)
    cancellation_ratio = torch.clamp(1.0 - (norm_error / (norm_actual + 1e-6)), min=0.0, max=1.0)
    mean_ratio = float(torch.mean(cancellation_ratio).item())

    return y_pred, e_discrepancy, mean_ratio


@triton.jit
def _forward_model_predictor_kernel(
    U_MOTOR_ptr,    # [B, D_MOTOR]
    Y_ACTUAL_ptr,   # [B, D_SENSORY]
    W_FWD_ptr,      # [D_SENSORY, D_MOTOR]
    Y_PRED_ptr,     # [B, D_SENSORY]
    E_DISC_ptr,     # [B, D_SENSORY]
    gain: float,
    B: tl.constexpr,
    D_MOTOR: tl.constexpr,
    D_SENSORY: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_b = offs_b < B
    mask_s = offs_s < D_SENSORY

    # Accumulate matrix multiplication over D_MOTOR
    acc = tl.zeros((BLOCK_B, BLOCK_S), dtype=tl.float32)

    for m in range(0, D_MOTOR, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < D_MOTOR

        # Load U_MOTOR: [BLOCK_B, BLOCK_M]
        u_ptrs = U_MOTOR_ptr + offs_b[:, None] * D_MOTOR + offs_m[None, :]
        u_val = tl.load(u_ptrs, mask=mask_b[:, None] & mask_m[None, :], other=0.0)

        # Load W_FWD: [BLOCK_S, BLOCK_M]
        w_ptrs = W_FWD_ptr + offs_s[:, None] * D_MOTOR + offs_m[None, :]
        w_val = tl.load(w_ptrs, mask=mask_s[:, None] & mask_m[None, :], other=0.0)

        # Matmul acc: [BLOCK_B, BLOCK_S] = u [B, M] @ w^T [M, S]
        acc += tl.dot(u_val, tl.trans(w_val))

    # Store predicted sensory reafference
    pred_ptrs = Y_PRED_ptr + offs_b[:, None] * D_SENSORY + offs_s[None, :]
    tl.store(pred_ptrs, acc, mask=mask_b[:, None] & mask_s[None, :])

    # Load actual sensory feedback
    act_ptrs = Y_ACTUAL_ptr + offs_b[:, None] * D_SENSORY + offs_s[None, :]
    y_act = tl.load(act_ptrs, mask=mask_b[:, None] & mask_s[None, :], other=0.0)

    # Discrepancy: e = actual - gain * pred
    e_disc = y_act - gain * acc
    disc_ptrs = E_DISC_ptr + offs_b[:, None] * D_SENSORY + offs_s[None, :]
    tl.store(disc_ptrs, e_disc, mask=mask_b[:, None] & mask_s[None, :])
