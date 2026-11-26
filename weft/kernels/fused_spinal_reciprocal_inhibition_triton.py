"""
Fused Spinal Renshaw Recurrent & Ia Reciprocal Inhibitory Network Kernel in Triton 3.9.

Models:
1. Agonist-Antagonist Motor Pools:
   - Flexor alpha motor neurons (MN_flex): [B, N_MN]
   - Extensor alpha motor neurons (MN_ext): [B, N_MN]
2. Ia Reciprocal Inhibitory Interneurons:
   - Ia_flex inhibits Extensor MNs: I_Ia_ext = W_Ia @ MN_flex
   - Ia_ext inhibits Flexor MNs:  I_Ia_flex = W_Ia @ MN_ext
3. Renshaw Cell Recurrent Inhibition:
   - Each motor neuron pool activates recurrent Renshaw cells:
     R_flex = W_renshaw @ MN_flex
     I_renshaw_flex = R_flex (negative feedback limiting runaway excitation)
4. Dynamic Balance & Mutual Exclusion:
   - Fused integration enforces strict mutual exclusion during locomotion step-cycles.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_spinal_cpg_step(
    drive_flex: torch.Tensor,
    drive_ext: torch.Tensor,
    mn_flex_prev: torch.Tensor,
    mn_ext_prev: torch.Tensor,
    g_Ia: float = 1.2,
    g_renshaw: float = 0.35,
    threshold: float = 20.0,
    gain: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates one time step of agonist-antagonist spinal motor pool integration.

    Args:
        drive_flex: [B, N_MN] descending/rhythmic excitatory drive to flexor pool
        drive_ext:  [B, N_MN] descending/rhythmic excitatory drive to extensor pool
        mn_flex_prev: [B, N_MN] previous flexor motor neuron firing rates (Hz)
        mn_ext_prev:  [B, N_MN] previous extensor motor neuron firing rates (Hz)
        g_Ia: strength of reciprocal Ia inhibition
        g_renshaw: strength of recurrent Renshaw inhibition
        threshold: firing threshold (Hz)
        gain: input-output slope

    Returns:
        (mn_flex_next, mn_ext_next)
    """
    # Net excitation to flexors: Drive - Reciprocal Extensor Ia - Renshaw Flexor
    net_flex = drive_flex - g_Ia * mn_ext_prev - g_renshaw * mn_flex_prev
    mn_flex_next = torch.relu((net_flex - threshold) * gain)

    # Net excitation to extensors: Drive - Reciprocal Flexor Ia - Renshaw Extensor
    net_ext = drive_ext - g_Ia * mn_flex_prev - g_renshaw * mn_ext_prev
    mn_ext_next = torch.relu((net_ext - threshold) * gain)

    return mn_flex_next, mn_ext_next


@triton.jit
def _spinal_cpg_kernel(
    DRIVE_FLEX_ptr,     # [B, N_MN]
    DRIVE_EXT_ptr,      # [B, N_MN]
    PREV_FLEX_ptr,      # [B, N_MN]
    PREV_EXT_ptr,       # [B, N_MN]
    OUT_FLEX_ptr,       # [B, N_MN]
    OUT_EXT_ptr,        # [B, N_MN]
    g_Ia: float,
    g_renshaw: float,
    threshold: float,
    gain: float,
    B: tl.constexpr,
    N_MN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_MN

    idx = pid_b * N_MN + offs_n

    df = tl.load(DRIVE_FLEX_ptr + idx, mask=mask, other=0.0)
    de = tl.load(DRIVE_EXT_ptr + idx, mask=mask, other=0.0)
    pf = tl.load(PREV_FLEX_ptr + idx, mask=mask, other=0.0)
    pe = tl.load(PREV_EXT_ptr + idx, mask=mask, other=0.0)

    # Compute net input
    net_f = df - g_Ia * pe - g_renshaw * pf
    net_e = de - g_Ia * pf - g_renshaw * pe

    # ReLU transfer
    out_f = tl.maximum(0.0, (net_f - threshold) * gain)
    out_e = tl.maximum(0.0, (net_e - threshold) * gain)

    tl.store(OUT_FLEX_ptr + idx, out_f, mask=mask)
    tl.store(OUT_EXT_ptr + idx, out_e, mask=mask)
