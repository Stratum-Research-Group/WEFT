"""
Cerebellar Parallel Fiber - Purkinje Cell Heterosynaptic Coincidence LTD Kernel in Triton 3.9.

Models:
1. Parallel Fiber (PF) Presynaptic Activity:
   x_PF in [B, N_PF]
2. Climbing Fiber (CF) Heterosynaptic Error Signal:
   e_CF in [B] (all-or-none error impulse / complex spike)
3. Intracellular Calcium & Second-Messenger Cascade:
   - PF stimulates AMPA & mGluR1 receptors -> diacylglycerol (DAG) + IP3
   - CF evokes massive dendritic Ca2+ influx via P/Q-type channels
   - Coincidence of elevated Ca2+ and DAG/IP3 triggers supra-additive PKCalpha activation
4. Long-Term Depression (LTD) Weight Update:
   dW_ij = - eta_LTD * (x_PF_j * e_CF_i) + eta_LTP * (x_PF_j * (1 - e_CF_i))
   W_ij = clamp(W_ij + dW_ij, W_min, W_max)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
from enum import IntEnum


class CerebellarPlasticityState(IntEnum):
    QUIESCENT = 0
    LTP_POTENTIATING = 1
    LTD_DEPRESSING = 2


def fused_cerebellar_ltd_step(
    weights: torch.Tensor,
    pf_activity: torch.Tensor,
    cf_error: torch.Tensor,
    eta_ltd: float = 0.05,
    eta_ltp: float = 0.005,
    w_min: float = 0.1,
    w_max: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates heterosynaptic coincidence LTD across parallel fiber synapses.

    Args:
        weights: [N_PC, N_PF] synaptic weights
        pf_activity: [B, N_PF] parallel fiber inputs
        cf_error: [B, N_PC] climbing fiber error signals (0 or 1)
        eta_ltd: rate of depression during coincidence
        eta_ltp: rate of potentiation during PF without CF
        w_min: minimum synaptic weight
        w_max: maximum synaptic weight

    Returns:
        (updated_weights, plasticity_state)
    """
    # Coincidence condition: PF active and CF active
    # pf_activity: [B, N_PF], cf_error: [B, N_PC]
    # For single batch or average:
    coincidence = torch.matmul(cf_error.t(), pf_activity) / max(pf_activity.shape[0], 1)
    pf_only = torch.matmul((1.0 - cf_error).t(), pf_activity) / max(pf_activity.shape[0], 1)

    delta_w = - eta_ltd * coincidence + eta_ltp * pf_only
    new_weights = torch.clamp(weights + delta_w, min=w_min, max=w_max)

    plasticity_state = torch.full_like(weights, CerebellarPlasticityState.QUIESCENT, dtype=torch.long)
    plasticity_state[coincidence > 0.1] = CerebellarPlasticityState.LTD_DEPRESSING
    plasticity_state[(pf_only > 0.2) & (coincidence <= 0.1)] = CerebellarPlasticityState.LTP_POTENTIATING

    return new_weights, plasticity_state


@triton.jit
def _cerebellar_ltd_kernel(
    W_ptr,          # [N_PC, N_PF]
    PF_ptr,         # [B, N_PF]
    CF_ptr,         # [B, N_PC]
    W_OUT_ptr,      # [N_PC, N_PF]
    eta_ltd: float,
    eta_ltp: float,
    w_min: float,
    w_max: float,
    B: tl.constexpr,
    N_PC: tl.constexpr,
    N_PF: tl.constexpr,
    BLOCK_PC: tl.constexpr,
    BLOCK_PF: tl.constexpr,
):
    pid_pc = tl.program_id(axis=0)
    pid_pf = tl.program_id(axis=1)

    offs_pc = pid_pc * BLOCK_PC + tl.arange(0, BLOCK_PC)
    offs_pf = pid_pf * BLOCK_PF + tl.arange(0, BLOCK_PF)

    mask_pc = offs_pc < N_PC
    mask_pf = offs_pf < N_PF

    # Accumulate coincidence and pf_only across batch
    acc_coinc = tl.zeros((BLOCK_PC, BLOCK_PF), dtype=tl.float32)
    acc_pfonley = tl.zeros((BLOCK_PC, BLOCK_PF), dtype=tl.float32)

    for b in range(B):
        # Load CF: [BLOCK_PC]
        cf_val = tl.load(CF_ptr + b * N_PC + offs_pc, mask=mask_pc, other=0.0)
        # Load PF: [BLOCK_PF]
        pf_val = tl.load(PF_ptr + b * N_PF + offs_pf, mask=mask_pf, other=0.0)

        # Outer product in block
        # [BLOCK_PC, 1] * [1, BLOCK_PF]
        cf_col = cf_val[:, None]
        pf_row = pf_val[None, :]

        acc_coinc += cf_col * pf_row
        acc_pfonley += (1.0 - cf_col) * pf_row

    scale = 1.0 / B
    delta_w = - eta_ltd * (acc_coinc * scale) + eta_ltp * (acc_pfonley * scale)

    # Load existing weight
    w_offs = offs_pc[:, None] * N_PF + offs_pf[None, :]
    w_mask = mask_pc[:, None] & mask_pf[None, :]
    curr_w = tl.load(W_ptr + w_offs, mask=w_mask, other=0.0)

    new_w = curr_w + delta_w
    new_w = tl.maximum(w_min, tl.minimum(w_max, new_w))

    tl.store(W_OUT_ptr + w_offs, new_w, mask=w_mask)
