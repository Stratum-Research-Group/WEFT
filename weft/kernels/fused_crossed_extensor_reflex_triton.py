"""
Fused Polysynaptic Flexor Reflex Afferent (FRA) & Crossed-Extensor Reflex Kernel in Triton 3.9.

Models:
1. Nociceptive Flexor Reflex Afferents (FRA):
   - High-threshold cutaneous A-delta & C fiber discharge: e_noxious [B, N_LIMBS]
2. Polysynaptic Ipsilateral Interneuronal Circuit:
   - Excitatory interneurons drive ipsilateral flexor alpha motor neurons (withdrawal)
   - Inhibitory interneurons shut down ipsilateral extensor alpha motor neurons:
     MN_ipsi_flex = relu(Drive_flex + w_fra_flex * e_noxious)
     MN_ipsi_ext  = relu(Drive_ext  - w_fra_ext_inhib * e_noxious)
3. Commissural Crossed-Extensor Circuit (V0/V3 Interneurons):
   - Axons decussate the anterior spinal commissure to the contralateral hemicord:
     Excite contralateral extensor motor pool (postural support thrust)
     Inhibit contralateral flexor motor pool:
     MN_contra_ext  = relu(Drive_ext + w_commissural_ext * e_noxious)
     MN_contra_flex = relu(Drive_flex - w_commissural_flex_inhib * e_noxious)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_crossed_extensor_step(
    e_noxious: torch.Tensor,
    drive_ipsi_flex: torch.Tensor,
    drive_ipsi_ext: torch.Tensor,
    drive_contra_flex: torch.Tensor,
    drive_contra_ext: torch.Tensor,
    w_ipsi_flex: float = 1.35,
    w_ipsi_ext_inhib: float = 1.50,
    w_contra_ext: float = 1.40,
    w_contra_flex_inhib: float = 1.50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates bilateral polysynaptic spinal reflex integration upon unilateral noxious stimulus.

    Args:
        e_noxious: [B, N_LIMBS] noxious stimulus intensity (0 = none, >50 = high pain)
        drive_ipsi_flex: [B, N_LIMBS] baseline descending flexor drive (ipsi)
        drive_ipsi_ext: [B, N_LIMBS] baseline descending extensor drive (ipsi)
        drive_contra_flex: [B, N_LIMBS] baseline descending flexor drive (contra)
        drive_contra_ext: [B, N_LIMBS] baseline descending extensor drive (contra)

    Returns:
        (mn_ipsi_flex, mn_ipsi_ext, mn_contra_ext, mn_contra_flex)
    """
    # Ipsilateral: Flexor excitation, extensor inhibition
    mn_ipsi_flex = torch.relu(drive_ipsi_flex + w_ipsi_flex * e_noxious)
    mn_ipsi_ext = torch.relu(drive_ipsi_ext - w_ipsi_ext_inhib * e_noxious)

    # Contralateral: Extensor excitation (thrust), flexor inhibition
    mn_contra_ext = torch.relu(drive_contra_ext + w_contra_ext * e_noxious)
    mn_contra_flex = torch.relu(drive_contra_flex - w_contra_flex_inhib * e_noxious)

    return mn_ipsi_flex, mn_ipsi_ext, mn_contra_ext, mn_contra_flex


@triton.jit
def _crossed_extensor_kernel(
    NOXIOUS_ptr,        # [B, N_LIMBS]
    IPSI_F_IN_ptr,      # [B, N_LIMBS]
    IPSI_E_IN_ptr,      # [B, N_LIMBS]
    CONTRA_F_IN_ptr,    # [B, N_LIMBS]
    CONTRA_E_IN_ptr,    # [B, N_LIMBS]
    IPSI_F_OUT_ptr,     # [B, N_LIMBS]
    IPSI_E_OUT_ptr,     # [B, N_LIMBS]
    CONTRA_E_OUT_ptr,   # [B, N_LIMBS]
    CONTRA_F_OUT_ptr,   # [B, N_LIMBS]
    w_if: float,
    w_ie: float,
    w_ce: float,
    w_cf: float,
    B: tl.constexpr,
    N_LIMBS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_LIMBS

    idx = pid_b * N_LIMBS + offs_n

    nox = tl.load(NOXIOUS_ptr + idx, mask=mask, other=0.0)
    i_f = tl.load(IPSI_F_IN_ptr + idx, mask=mask, other=0.0)
    i_e = tl.load(IPSI_E_IN_ptr + idx, mask=mask, other=0.0)
    c_f = tl.load(CONTRA_F_IN_ptr + idx, mask=mask, other=0.0)
    c_e = tl.load(CONTRA_E_IN_ptr + idx, mask=mask, other=0.0)

    # Compute outputs
    out_if = tl.maximum(0.0, i_f + w_if * nox)
    out_ie = tl.maximum(0.0, i_e - w_ie * nox)
    out_ce = tl.maximum(0.0, c_e + w_ce * nox)
    out_cf = tl.maximum(0.0, c_f - w_cf * nox)

    tl.store(IPSI_F_OUT_ptr + idx, out_if, mask=mask)
    tl.store(IPSI_E_OUT_ptr + idx, out_ie, mask=mask)
    tl.store(CONTRA_E_OUT_ptr + idx, out_ce, mask=mask)
    tl.store(CONTRA_F_OUT_ptr + idx, out_cf, mask=mask)
