"""
Fused Magnocellular Red Nucleus & Rubrospinal Velocity Control Kernel in Triton 3.9.

Models:
1. Interposed Cerebellar Nucleus & Motor Cortical Convergence:
   - Magnocellular Red Nucleus (RNm) receives somatotopically organized inputs:
     Branch A: Deep cerebellar interposed nucleus (emboliform + globose): x_inter [B, N_UNITS]
     Branch B: Primary motor cortex (M1 layer V corticorubral): x_m1 [B, N_UNITS]
2. Instantaneous Limb Velocity Gating:
   - RNm firing rate scales linearly with velocity of distal flexor execution:
     f_RNm = clamp(base_rnm + k_vel * v_limb + w_inter * x_inter + w_m1 * x_m1, 0, f_max)
3. Rubrospinal Tract (RST) Decussation & Distal Flexor Facilitation:
   - Ventral tegmental decussation of Forel (axons cross midbrain midline).
   - Descends in lateral funiculus to cervical enlargement (laminae V-VII).
   - Monosynaptically excites distal digit flexor motoneurons while inhibiting extensors:
     MN_digit_flex = relu(base_flex + w_rst * f_RNm)
4. Tremor Dampening:
   - Closed-loop cerebello-rubro-olivary feedback suppresses hypermetric oscillation (intention tremor).
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_rubrospinal_velocity_step(
    v_limb: torch.Tensor,
    x_inter: torch.Tensor,
    x_m1: torch.Tensor,
    base_rnm: float = 35.0,
    k_vel: float = 85.0,
    w_inter: float = 0.55,
    w_m1: float = 0.40,
    w_rst: float = 1.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates red nucleus magnocellular velocity integration and rubrospinal digit flexor recruitment.

    Args:
        v_limb: [B, N_UNITS] instantaneous distal limb velocity (m/s)
        x_inter: [B, N_UNITS] cerebellar interposed nuclear drive (Hz)
        x_m1: [B, N_UNITS] primary motor cortical corticorubral drive (Hz)

    Returns:
        (f_RNm, mn_digit_flex)
    """
    v_pos = torch.clamp(v_limb, min=0.0)

    # RNm discharge rate
    f_RNm = torch.clamp(
        base_rnm + k_vel * v_pos + w_inter * x_inter + w_m1 * x_m1,
        min=0.0,
        max=300.0
    )

    # Distal flexor alpha motor neuron drive
    mn_digit_flex = torch.clamp(20.0 + w_rst * (f_RNm - base_rnm), min=0.0, max=250.0)

    return f_RNm, mn_digit_flex


@triton.jit
def _rubrospinal_velocity_kernel(
    V_LIMB_ptr,         # [B, N_UNITS]
    X_INTER_ptr,        # [B, N_UNITS]
    X_M1_ptr,           # [B, N_UNITS]
    RNM_OUT_ptr,        # [B, N_UNITS]
    FLEX_OUT_ptr,       # [B, N_UNITS]
    base_rnm: float,
    k_vel: float,
    w_inter: float,
    w_m1: float,
    w_rst: float,
    B: tl.constexpr,
    N_UNITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_UNITS

    idx = pid_b * N_UNITS + offs_n

    v = tl.load(V_LIMB_ptr + idx, mask=mask, other=0.0)
    x_in = tl.load(X_INTER_ptr + idx, mask=mask, other=0.0)
    x_m = tl.load(X_M1_ptr + idx, mask=mask, other=0.0)

    v_pos = tl.maximum(0.0, v)
    rnm = tl.maximum(0.0, tl.minimum(300.0, base_rnm + k_vel * v_pos + w_inter * x_in + w_m1 * x_m))

    flex = tl.maximum(0.0, tl.minimum(250.0, 20.0 + w_rst * (rnm - base_rnm)))

    tl.store(RNM_OUT_ptr + idx, rnm, mask=mask)
    tl.store(FLEX_OUT_ptr + idx, flex, mask=mask)
