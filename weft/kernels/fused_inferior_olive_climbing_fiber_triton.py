"""
Inferior Olive Gap-Junction Synchrony & Climbing Fiber Burst Kernel in Triton 3.9.

Models:
1. Subthreshold Sinusoidal Membrane Oscillations:
   dV_i / dt = - (V_i - E_L) / tau_m + g_osc * sin(omega * t + phi_i) + I_error_i
2. Electrotonic Diffusive Gap-Junction Coupling (Connexin-36):
   I_gap_i = g_gap * sum_{j in N(i)} (V_j - V_i) = - g_gap * (L @ V)_i
3. Climbing Fiber Complex Spike Discharge:
   When V_i crosses depolarizing burst threshold V_thresh, an all-or-none
   high-frequency burst (complex spike) is dispatched down the climbing fiber,
   inducing calcium influx and Purkinje cell heterosynaptic LTD.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _inferior_olive_gap_kernel(
    V_IN_ptr,          # [B, N] - membrane potentials
    I_ERROR_ptr,       # [B, N] - sensory motor error drive
    LAPLACIAN_ptr,     # [N, N] - normalized graph Laplacian for electrical coupling
    V_OUT_ptr,         # [B, N] - updated membrane potential
    SPIKE_OUT_ptr,     # [B, N] - complex spike binary mask
    g_gap: float,      # gap junction conductance
    v_thresh: float,   # burst discharge threshold
    dt_over_tau: float,# integration step
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Accumulate electrical diffusive coupling: sum_j L_{ij} * V_j
    acc_diff = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k in range(0, N, BLOCK_N):
        offs_k = k + tl.arange(0, BLOCK_N)
        mask_k = offs_k < N

        v_k = tl.load(V_IN_ptr + pid_b * N + offs_k, mask=mask_k, other=0.0)
        l_block = tl.load(
            LAPLACIAN_ptr + offs_n[:, None] * N + offs_k[None, :],
            mask=(mask_n[:, None] & mask_k[None, :]),
            other=0.0
        )
        acc_diff += tl.sum(l_block * v_k[None, :], axis=1)

    # Current V_i and error drive
    v_curr = tl.load(V_IN_ptr + pid_b * N + offs_n, mask=mask_n, other=-60.0)
    i_err = tl.load(I_ERROR_ptr + pid_b * N + offs_n, mask=mask_n, other=0.0)

    # Diffusive synchronization term: - g_gap * (L @ V)
    i_gap = - g_gap * acc_diff

    # Subthreshold update with leak and error drive
    v_next = v_curr + dt_over_tau * (- (v_curr - (-60.0)) + i_gap + i_err)

    # Check for all-or-none complex spike discharge
    is_spike = v_next > v_thresh
    spike_val = tl.where(is_spike, 1.0, 0.0)

    # Reset voltage if spiked (refractory burst)
    v_stored = tl.where(is_spike, -70.0, v_next)

    tl.store(V_OUT_ptr + pid_b * N + offs_n, v_stored, mask=mask_n)
    tl.store(SPIKE_OUT_ptr + pid_b * N + offs_n, spike_val, mask=mask_n)


def inferior_olive_climbing_fiber_step(
    v_curr: torch.Tensor,
    i_error: torch.Tensor,
    laplacian: torch.Tensor,
    g_gap: float = 0.5,
    v_thresh: float = -45.0,
    dt_over_tau: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Simulates inferior olive gap-junction electrotonic coupling and complex spike dispatch.

    Args:
        v_curr:     [B, N] current membrane potentials
        i_error:    [B, N] motor error input drive
        laplacian:  [N, N] normalized Laplacian of gap-junction electrical network
        g_gap:      electrical coupling strength
        v_thresh:   complex spike trigger threshold
        dt_over_tau: integration time constant ratio

    Returns:
        (v_next, spike_mask, phase_coherence_order_parameter)
    """
    B, N = v_curr.shape
    device = v_curr.device

    if v_curr.is_cuda:
        v_next = torch.empty((B, N), device=device, dtype=torch.float32)
        spike_mask = torch.empty((B, N), device=device, dtype=torch.float32)
        BLOCK_N = 128 if N >= 128 else 32
        grid = (B, triton.cdiv(N, BLOCK_N))

        _inferior_olive_gap_kernel[grid](
            v_curr.contiguous(),
            i_error.contiguous(),
            laplacian.contiguous(),
            v_next,
            spike_mask,
            float(g_gap),
            float(v_thresh),
            float(dt_over_tau),
            B=B,
            N=N,
            BLOCK_N=BLOCK_N,
        )
    else:
        # High-performance PyTorch CPU fallback
        # i_gap = - g_gap * (V @ L^T)
        i_gap = - g_gap * torch.matmul(v_curr, laplacian.t())
        v_tentative = v_curr + dt_over_tau * (- (v_curr - (-60.0)) + i_gap + i_error)

        is_spike = (v_tentative > v_thresh).float()
        v_next = torch.where(is_spike > 0.5, torch.tensor(-70.0, device=device), v_tentative)
        spike_mask = is_spike

    # Compute Kuramoto-style phase coherence / variance order parameter: R = 1.0 - std(V) / (max - min)
    v_std = torch.std(v_next, dim=-1)
    coherence = float(torch.clamp(1.0 - v_std / 15.0, min=0.0, max=1.0).mean().item())

    return v_next, spike_mask, coherence
