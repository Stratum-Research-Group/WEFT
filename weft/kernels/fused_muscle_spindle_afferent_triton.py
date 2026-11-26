"""
Fused Muscle Spindle Afferent Discharge & Monosynaptic Reflex Kernel in Triton 3.9.

Models:
1. Intrafusal Fiber Mechanics:
   - Nuclear bag1 fibers (dynamic sensitivity, velocity-dependent)
   - Nuclear bag2 & chain fibers (static sensitivity, length-dependent)
2. Primary Group Ia Afferent Discharge:
   - F_Ia = clamp(k_pos * (L - L_0) + k_vel * sign(dL/dt) * |dL/dt|^0.6 * (1 + gamma_dyn), 0, F_max)
3. Secondary Group II Afferent Discharge:
   - F_II = clamp(k_pos_II * (L - L_0) * (1 + gamma_stat), 0, F_max)
4. Monosynaptic Alpha Motor Neuron EPSP & Reflex Discharge:
   - Direct homonymous alpha motor neuron recruitment:
     MN_reflex = relu((w_Ia_to_MN * F_Ia - theta_reflex) * gain_reflex)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_muscle_spindle_reflex_step(
    muscle_length: torch.Tensor,
    stretch_velocity: torch.Tensor,
    gamma_dynamic: torch.Tensor,
    gamma_static: torch.Tensor,
    l_0: float = 1.0,
    k_pos_Ia: float = 80.0,
    k_vel_Ia: float = 95.0,
    k_pos_II: float = 70.0,
    w_Ia_to_mn: float = 1.35,
    theta_reflex: float = 40.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates muscle spindle Ia/II afferent discharge and monosynaptic alpha reflex recruitment.

    Args:
        muscle_length: [B, N_SPINDLES] normalized muscle length (L / L0)
        stretch_velocity: [B, N_SPINDLES] lengthening velocity (dL/dt)
        gamma_dynamic: [B, N_SPINDLES] dynamic fusimotor drive [0, 1]
        gamma_static:  [B, N_SPINDLES] static fusimotor drive [0, 1]

    Returns:
        (f_Ia, f_II, alpha_mn_reflex)
    """
    delta_L = torch.clamp(muscle_length - l_0, min=-0.3, max=0.8)
    v_pos = torch.clamp(stretch_velocity, min=0.0)

    # Nonlinear power-law velocity sensitivity (Matthews 1972)
    v_comp = torch.pow(v_pos + 1e-6, 0.6)

    # Group Ia afferent firing rate (Hz)
    f_Ia = torch.clamp(35.0 + k_pos_Ia * delta_L + k_vel_Ia * v_comp * (1.0 + gamma_dynamic), min=0.0, max=500.0)

    # Group II afferent firing rate (Hz)
    f_II = torch.clamp(25.0 + k_pos_II * delta_L * (1.0 + gamma_static), min=0.0, max=300.0)

    # Monosynaptic reflex drive to homonymous alpha motor pool
    alpha_mn_reflex = torch.relu((f_Ia - theta_reflex) * w_Ia_to_mn)

    return f_Ia, f_II, alpha_mn_reflex


@triton.jit
def _spindle_reflex_kernel(
    L_ptr,              # [B, N_SPINDLES]
    V_ptr,              # [B, N_SPINDLES]
    G_DYN_ptr,          # [B, N_SPINDLES]
    G_STAT_ptr,         # [B, N_SPINDLES]
    F_IA_OUT_ptr,       # [B, N_SPINDLES]
    F_II_OUT_ptr,       # [B, N_SPINDLES]
    MN_REFLEX_OUT_ptr,  # [B, N_SPINDLES]
    l_0: float,
    k_pos_Ia: float,
    k_vel_Ia: float,
    k_pos_II: float,
    w_Ia: float,
    theta_ref: float,
    B: tl.constexpr,
    N_SPINDLES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_SPINDLES

    idx = pid_b * N_SPINDLES + offs_n

    L = tl.load(L_ptr + idx, mask=mask, other=1.0)
    v = tl.load(V_ptr + idx, mask=mask, other=0.0)
    g_dyn = tl.load(G_DYN_ptr + idx, mask=mask, other=0.0)
    g_stat = tl.load(G_STAT_ptr + idx, mask=mask, other=0.0)

    dL = L - l_0
    v_pos = tl.maximum(0.0, v)

    # Approximate v^0.6 via sqrt * sqrt(sqrt)
    # v^0.5 * v^0.125 ~ v^0.625
    v_half = tl.sqrt(v_pos)
    v_pow = tl.sqrt(tl.sqrt(v_half)) * v_half

    f_ia = tl.maximum(0.0, tl.minimum(500.0, 35.0 + k_pos_Ia * dL + k_vel_Ia * v_pow * (1.0 + g_dyn)))
    f_ii = tl.maximum(0.0, tl.minimum(300.0, 25.0 + k_pos_II * dL * (1.0 + g_stat)))

    mn_ref = tl.maximum(0.0, (f_ia - theta_ref) * w_Ia)

    tl.store(F_IA_OUT_ptr + idx, f_ia, mask=mask)
    tl.store(F_II_OUT_ptr + idx, f_ii, mask=mask)
    tl.store(MN_REFLEX_OUT_ptr + idx, mn_ref, mask=mask)
