"""
Prefrontal Working Memory Persistent Delay & Mediodorsal (MD) Thalamocortical Gating Kernel in Triton 3.9.

Models:
1. Reciprocal Prefrontal - Mediodorsal Thalamic positive feedback loop:
   tau_pfc * dx_pfc / dt = -x_pfc + f(W_pfc @ x_pfc + gamma_md * (W_th_to_ctx @ x_md) + I_cue - I_inh)
   tau_md  * dx_md  / dt = -x_md  + f(W_ctx_to_th @ x_pfc)
2. Bistable Attractor Dynamics:
   - Low-rate baseline state (~2-5 Hz)
   - High-rate persistent delay state (~40-60 Hz) representing maintained working memory items.
3. Distractor Suppression:
   MD thalamic gain gamma_md amplifies signal-to-noise ratio, shielding delay activity from transient distractors.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _pfc_md_delay_step_kernel(
    X_PFC_ptr,          # [B, N_PFC]
    X_MD_ptr,           # [B, N_MD]
    W_PFC_ptr,          # [N_PFC, N_PFC]
    W_TH_TO_CTX_ptr,    # [N_PFC, N_MD]
    W_CTX_TO_TH_ptr,    # [N_MD, N_PFC]
    X_PFC_NEXT_ptr,     # [B, N_PFC]
    X_MD_NEXT_ptr,      # [B, N_MD]
    gamma_md: float,    # MD thalamic gain parameter
    dt_over_tau: float, # discrete integration step
    inh_bias: float,    # recurrent inhibitory threshold
    B: tl.constexpr,
    N_PFC: tl.constexpr,
    N_MD: tl.constexpr,
    BLOCK_PFC: tl.constexpr,
    BLOCK_MD: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_p = tl.program_id(axis=1)

    offs_p = pid_p * BLOCK_PFC + tl.arange(0, BLOCK_PFC)
    mask_p = offs_p < N_PFC

    # Accumulate recurrent PFC excitation: W_pfc @ x_pfc
    acc_pfc = tl.zeros([BLOCK_PFC], dtype=tl.float32)
    for k in range(0, N_PFC, BLOCK_PFC):
        offs_k = k + tl.arange(0, BLOCK_PFC)
        mask_k = offs_k < N_PFC
        x_k = tl.load(X_PFC_ptr + pid_b * N_PFC + offs_k, mask=mask_k, other=0.0)
        w_block = tl.load(
            W_PFC_ptr + offs_p[:, None] * N_PFC + offs_k[None, :],
            mask=(mask_p[:, None] & mask_k[None, :]),
            other=0.0
        )
        acc_pfc += tl.sum(w_block * x_k[None, :], axis=1)

    # Accumulate thalamic feedback: W_th_to_ctx @ x_md
    acc_th = tl.zeros([BLOCK_PFC], dtype=tl.float32)
    for k in range(0, N_MD, BLOCK_MD):
        offs_k = k + tl.arange(0, BLOCK_MD)
        mask_k = offs_k < N_MD
        x_md_k = tl.load(X_MD_ptr + pid_b * N_MD + offs_k, mask=mask_k, other=0.0)
        w_th = tl.load(
            W_TH_TO_CTX_ptr + offs_p[:, None] * N_MD + offs_k[None, :],
            mask=(mask_p[:, None] & mask_k[None, :]),
            other=0.0
        )
        acc_th += tl.sum(w_th * x_md_k[None, :], axis=1)

    # Current PFC state
    x_pfc_curr = tl.load(X_PFC_ptr + pid_b * N_PFC + offs_p, mask=mask_p, other=0.0)

    # Non-linear activation: relu(acc_pfc + gamma_md * acc_th - inh_bias)
    drive_pfc = acc_pfc + gamma_md * acc_th - inh_bias
    act_pfc = tl.maximum(drive_pfc, 0.0)

    # Euler integration step for PFC
    x_pfc_next = x_pfc_curr + dt_over_tau * (-x_pfc_curr + act_pfc)
    tl.store(X_PFC_NEXT_ptr + pid_b * N_PFC + offs_p, x_pfc_next, mask=mask_p)


def pfc_md_working_memory_step(
    x_pfc: torch.Tensor,
    x_md: torch.Tensor,
    w_pfc: torch.Tensor,
    w_th_to_ctx: torch.Tensor,
    w_ctx_to_th: torch.Tensor,
    gamma_md: float = 2.5,
    dt_over_tau: float = 0.1,
    inh_bias: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs one discrete delay-period step of the reciprocal PFC-MD thalamic loop.

    Args:
        x_pfc: [B, N_PFC] current prefrontal rate
        x_md:  [B, N_MD]  current mediodorsal thalamic rate
        w_pfc: [N_PFC, N_PFC] recurrent cortical connections
        w_th_to_ctx: [N_PFC, N_MD] thalamocortical projection
        w_ctx_to_th: [N_MD, N_PFC] corticothalamic projection
        gamma_md: thalamic gain
        dt_over_tau: integration step
        inh_bias: global inhibitory threshold

    Returns:
        (x_pfc_next, x_md_next)
    """
    B, N_PFC = x_pfc.shape
    _, N_MD = x_md.shape

    if x_pfc.is_cuda:
        x_pfc_next = torch.empty_like(x_pfc)
        x_md_next = torch.empty_like(x_md)

        BLOCK_PFC = 128 if N_PFC >= 128 else 32
        BLOCK_MD = 64 if N_MD >= 64 else 16
        grid = (B, triton.cdiv(N_PFC, BLOCK_PFC))

        _pfc_md_delay_step_kernel[grid](
            x_pfc.contiguous(),
            x_md.contiguous(),
            w_pfc.contiguous(),
            w_th_to_ctx.contiguous(),
            w_ctx_to_th.contiguous(),
            x_pfc_next,
            x_md_next,
            float(gamma_md),
            float(dt_over_tau),
            float(inh_bias),
            B=B,
            N_PFC=N_PFC,
            N_MD=N_MD,
            BLOCK_PFC=BLOCK_PFC,
            BLOCK_MD=BLOCK_MD,
        )

        # Update MD thalamus: x_md + dt * (-x_md + relu(x_pfc @ W^T))
        drive_md = torch.relu(torch.matmul(x_pfc, w_ctx_to_th.t()))
        x_md_next = x_md + dt_over_tau * (-x_md + drive_md)
        return x_pfc_next, x_md_next
    else:
        # High-performance PyTorch CPU fallback
        drive_pfc = torch.matmul(x_pfc, w_pfc.t()) + gamma_md * torch.matmul(x_md, w_th_to_ctx.t()) - inh_bias
        act_pfc = torch.relu(drive_pfc)
        x_pfc_next = x_pfc + dt_over_tau * (-x_pfc + act_pfc)

        drive_md = torch.relu(torch.matmul(x_pfc, w_ctx_to_th.t()))
        x_md_next = x_md + dt_over_tau * (-x_md + drive_md)
        return x_pfc_next, x_md_next
