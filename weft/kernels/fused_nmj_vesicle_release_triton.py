"""
Fused Neuromuscular Junction (NMJ) Quantal ACh Release & Endplate Potential (EPP) Kernel in Triton 3.9.

Models:
1. Nerve Terminal Action Potential & Presynaptic Ca2+ Influx:
   - Ca2+ cooperativity: Release probability P_rel propto [Ca2+]^4
   - Quantal vesicle content m = n * p (where n ~ 300 readily releasable pool vesicles, p ~ 0.2)
2. Acetylcholine (ACh) Junctional Diffusion & AChE Clearance:
   - Cleft [ACh] peaks at ~1.0 mM and is hydrolyzed in <0.5 ms by acetylcholinesterase
3. Nicotinic Acetylcholine Receptor (nAChR) Channel Gating:
   - Reversal potential E_rev ~ 0 mV
   - EPP amplitude: Delta V = m * v_1 (where v_1 ~ 0.7 mV per quantum)
4. Safety Factor & Muscle Action Potential All-or-None Thresholding:
   - EPP > V_thresh (typically ~25 mV depolarization from -85 mV to -60 mV)
   - Safety factor SF = EPP / V_thresh (normally 1.5 - 2.0x in mammalian muscle)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_nmj_epp_step(
    ca_influx: torch.Tensor,
    readily_releasable_pool: torch.Tensor,
    v_muscle_rest: torch.Tensor,
    v_thresh_depol: float = 25.0,
    quantum_amplitude: float = 0.70,
    cooperativity_pow: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates NMJ quantal transmitter release and endplate potential depolarization.

    Args:
        ca_influx: [B, N_FIBERS] normalized presynaptic calcium transient [0, 1]
        readily_releasable_pool: [B, N_FIBERS] available vesicle pool (e.g. ~300)
        v_muscle_rest: [B, N_FIBERS] resting sarcolemmal potential (mV, e.g. -85.0)
        v_thresh_depol: depolarization needed to reach voltage-gated Na+ threshold (mV, e.g. 25.0)
        quantum_amplitude: single miniature EPP (MEPP) size in mV (~0.7 mV)
        cooperativity_pow: calcium cooperativity exponent (4.0)

    Returns:
        (quanta_released, epp_amplitude, muscle_ap_fired)
    """
    # 4th-power calcium cooperativity
    p_release = torch.clamp(torch.pow(ca_influx, cooperativity_pow) * 0.45, min=0.001, max=0.85)

    # Number of quanta released
    quanta_released = readily_releasable_pool * p_release

    # Endplate Potential amplitude (mV)
    epp_amplitude = quanta_released * quantum_amplitude

    # Muscle action potential elicited if EPP exceeds threshold depolarization
    muscle_ap_fired = (epp_amplitude >= v_thresh_depol).float()

    return quanta_released, epp_amplitude, muscle_ap_fired


@triton.jit
def _nmj_quantal_kernel(
    CA_INFLUX_ptr,      # [B, N_FIBERS]
    POOL_ptr,           # [B, N_FIBERS]
    QUANTA_OUT_ptr,     # [B, N_FIBERS]
    EPP_OUT_ptr,        # [B, N_FIBERS]
    AP_OUT_ptr,         # [B, N_FIBERS]
    v_thresh: float,
    quantum_amp: float,
    B: tl.constexpr,
    N_FIBERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_FIBERS

    idx = pid_b * N_FIBERS + offs_n

    ca = tl.load(CA_INFLUX_ptr + idx, mask=mask, other=0.0)
    pool = tl.load(POOL_ptr + idx, mask=mask, other=0.0)

    # 4th-power Ca cooperativity: ca^4
    ca2 = ca * ca
    ca4 = ca2 * ca2
    p_rel = tl.maximum(0.001, tl.minimum(0.85, ca4 * 0.45))

    quanta = pool * p_rel
    epp = quanta * quantum_amp

    fired = tl.where(epp >= v_thresh, 1.0, 0.0)

    tl.store(QUANTA_OUT_ptr + idx, quanta, mask=mask)
    tl.store(EPP_OUT_ptr + idx, epp, mask=mask)
    tl.store(AP_OUT_ptr + idx, fired, mask=mask)
