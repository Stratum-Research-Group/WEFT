"""
Fused Golgi Tendon Organ (GTO) Ib Force Feedback & Autogenic Inhibition Kernel in Triton 3.9.

Models:
1. In-Series Tendon Mechanotransduction:
   - GTOs sit in series with extrafusal muscle fibers at the myotendinous junction.
   - Collagen strand deformation compresses unmyelinated Group Ib axon terminals.
   - Discharge rate follows power-law force scaling:
     F_Ib = clamp(k_Ib * (Tension)^0.85, 0, F_max)
2. Di-synaptic Glycinergic Autogenic Inhibition:
   - Group Ib afferents enter dorsal horn and excite spinal Ib inhibitory interneurons.
   - Ib interneurons release glycine onto homonymous alpha motor neurons:
     I_autogenic = w_Ib_to_IN * F_Ib
3. Clasp-Knife Tendon Overload Protection:
   - When tension exceeds critical threshold (T_critical), autogenic inhibition drives
     total alpha motoneuronal cessation to prevent tendon avulsion:
     MN_drive = where(Tension >= T_critical, 0.0, relu(Drive_descending - I_autogenic))
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_gto_force_inhibition_step(
    muscle_tension: torch.Tensor,
    descending_drive: torch.Tensor,
    k_Ib: float = 4.5,
    t_critical: float = 120.0,
    w_Ib_to_in: float = 0.55,
    f_max_Ib: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates GTO force transduction, Ib interneuron autogenic inhibition, and protective clasp-knife release.

    Args:
        muscle_tension: [B, N_TENDONS] active tensile force (Newtons)
        descending_drive: [B, N_TENDONS] supraspinal excitatory drive to alpha motor pool (Hz)
        k_Ib: force-to-discharge gain coefficient
        t_critical: critical force ceiling eliciting clasp-knife shutoff
        w_Ib_to_in: synaptic weight of autogenic inhibition
        f_max_Ib: saturation ceiling for Ib afferents

    Returns:
        (f_Ib, autogenic_inhibition, net_alpha_drive)
    """
    t_pos = torch.clamp(muscle_tension, min=0.0)

    # Power-law force transduction: T^0.85
    f_Ib = torch.clamp(k_Ib * torch.pow(t_pos + 1e-5, 0.85), min=0.0, max=f_max_Ib)

    # Di-synaptic autogenic inhibition
    autogenic_inhibition = w_Ib_to_in * f_Ib

    # Clasp-knife condition: if tension >= t_critical, force complete motor drop
    is_overloaded = muscle_tension >= t_critical
    net_alpha_drive = torch.where(
        is_overloaded,
        torch.zeros_like(descending_drive),
        torch.relu(descending_drive - autogenic_inhibition)
    )

    return f_Ib, autogenic_inhibition, net_alpha_drive


@triton.jit
def _gto_force_kernel(
    TENSION_ptr,        # [B, N_TENDONS]
    DRIVE_ptr,          # [B, N_TENDONS]
    F_IB_OUT_ptr,       # [B, N_TENDONS]
    INHIB_OUT_ptr,      # [B, N_TENDONS]
    ALPHA_OUT_ptr,      # [B, N_TENDONS]
    k_Ib: float,
    t_critical: float,
    w_Ib: float,
    f_max: float,
    B: tl.constexpr,
    N_TENDONS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_TENDONS

    idx = pid_b * N_TENDONS + offs_n

    tension = tl.load(TENSION_ptr + idx, mask=mask, other=0.0)
    drive = tl.load(DRIVE_ptr + idx, mask=mask, other=0.0)

    t_pos = tl.maximum(0.0, tension)

    # Fast power-law approximation T^0.85 ~ T * (T)^(-0.15) or sqrt(T) * sqrt(sqrt(T))
    # T^0.5 * T^0.25 * T^0.125 = T^0.875 ~ T^0.85
    t_half = tl.sqrt(t_pos)
    t_quarter = tl.sqrt(t_half)
    t_eighth = tl.sqrt(t_quarter)
    t_pow = t_half * t_quarter * t_eighth

    f_ib = tl.maximum(0.0, tl.minimum(f_max, k_Ib * t_pow))
    inhib = w_Ib * f_ib

    # Clasp-knife check
    is_over = tension >= t_critical
    alpha = tl.where(is_over, 0.0, tl.maximum(0.0, drive - inhib))

    tl.store(F_IB_OUT_ptr + idx, f_ib, mask=mask)
    tl.store(INHIB_OUT_ptr + idx, inhib, mask=mask)
    tl.store(ALPHA_OUT_ptr + idx, alpha, mask=mask)
