"""
Fused Pontomedullary Reticulospinal Tract (PRST/MRST) Anticipatory Feedforward Postural Kernel in Triton 3.9.

Models:
1. Premotor Cortex (PMC) & Supplementary Motor Area (SMA) Intentional Drive:
   - Issues parallel motor plan branching:
     Branch A: Reticulospinal APA command to Pontomedullary Reticular Formation (PMRF)
     Branch B: Delayed Corticospinal command to M1 focal effectors (lead time Delta t = 60-100 ms)
2. Pontomedullary Reticulospinal Tract (PRST / MRST) Bilateral Divergence:
   - Pontine reticulospinal tract (PRST): Ipsilateral uncrossed descending drive to axial/limb extensors
   - Medullary reticulospinal tract (MRST): Bilateral inhibitory/excitatory modulation of spinal interneurons
   - Synchronous feedforward recruitment of gastrocnemius, soleus, and lumbar paraspinal muscles:
     APA_drive = relu(base_rst + w_pmc_rst * drive_intent)
3. Corticospinal Focal Effector Execution:
   - Focal voluntary limb movement (e.g. deltoid shoulder flexion):
     Focal_drive = where(t >= t_onset_focal, relu(base_cst + w_m1_cst * drive_intent), 0.0)
4. Reactive Torque Cancellation:
   - Anticipatory postural torque cancels reactive recoil momentum:
     Net_Postural_Disruption = |Torque_reactive - Torque_apa|
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_reticulospinal_apa_step(
    drive_intent: torch.Tensor,
    t_relative_ms: torch.Tensor,
    apa_lead_ms: float = 75.0,
    base_rst: float = 30.0,
    w_pmc_rst: float = 1.25,
    w_m1_cst: float = 1.40,
    torque_focal_gain: float = 0.85,
    torque_apa_gain: float = 0.82,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates reticulospinal anticipatory postural adjustment and focal motor execution.

    Args:
        drive_intent: [B, N_SYNERGIES] premotor voluntary intentional command [0, 100]
        t_relative_ms: [B, N_SYNERGIES] time relative to movement plan onset (-100 to +300 ms)
        apa_lead_ms: feedforward lead time for APA before focal onset (ms, e.g. 75 ms)

    Returns:
        (apa_reticulospinal_drive, focal_corticospinal_drive, net_postural_disruption)
    """
    # APA triggers immediately upon movement planning (t >= -apa_lead_ms)
    is_apa_time = t_relative_ms >= -apa_lead_ms
    apa_reticulospinal_drive = torch.where(
        is_apa_time & (drive_intent > 20.0),
        torch.clamp(base_rst + w_pmc_rst * drive_intent, min=0.0, max=250.0),
        torch.full_like(drive_intent, base_rst)
    )

    # Focal movement executes at t >= 0.0 ms
    is_focal_time = t_relative_ms >= 0.0
    focal_corticospinal_drive = torch.where(
        is_focal_time & (drive_intent > 20.0),
        torch.clamp(10.0 + w_m1_cst * drive_intent, min=0.0, max=250.0),
        torch.full_like(drive_intent, 10.0)
    )

    # Reactive torque vs APA stabilizing torque
    t_reactive = focal_corticospinal_drive * torque_focal_gain
    t_stabilizing = (apa_reticulospinal_drive - base_rst) * torque_apa_gain
    net_postural_disruption = torch.abs(t_reactive - t_stabilizing)

    return apa_reticulospinal_drive, focal_corticospinal_drive, net_postural_disruption


@triton.jit
def _reticulospinal_apa_kernel(
    INTENT_ptr,         # [B, N_SYNERGIES]
    TIME_ptr,           # [B, N_SYNERGIES]
    APA_OUT_ptr,        # [B, N_SYNERGIES]
    FOCAL_OUT_ptr,      # [B, N_SYNERGIES]
    DISRUPT_OUT_ptr,    # [B, N_SYNERGIES]
    apa_lead: float,
    base_rst: float,
    w_rst: float,
    w_cst: float,
    t_focal_gain: float,
    t_apa_gain: float,
    B: tl.constexpr,
    N_SYNERGIES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_SYNERGIES

    idx = pid_b * N_SYNERGIES + offs_n

    intent = tl.load(INTENT_ptr + idx, mask=mask, other=0.0)
    t_rel = tl.load(TIME_ptr + idx, mask=mask, other=0.0)

    # APA condition: t_rel >= -apa_lead and intent > 20
    is_apa = (t_rel >= -apa_lead) & (intent > 20.0)
    apa_drive = tl.where(is_apa, tl.minimum(250.0, base_rst + w_rst * intent), base_rst)

    # Focal condition: t_rel >= 0.0 and intent > 20
    is_focal = (t_rel >= 0.0) & (intent > 20.0)
    focal_drive = tl.where(is_focal, tl.minimum(250.0, 10.0 + w_cst * intent), 10.0)

    t_reac = focal_drive * t_focal_gain
    t_stab = (apa_drive - base_rst) * t_apa_gain
    diff = t_reac - t_stab
    disrupt = tl.abs(diff)

    tl.store(APA_OUT_ptr + idx, apa_drive, mask=mask)
    tl.store(FOCAL_OUT_ptr + idx, focal_drive, mask=mask)
    tl.store(DISRUPT_OUT_ptr + idx, disrupt, mask=mask)
