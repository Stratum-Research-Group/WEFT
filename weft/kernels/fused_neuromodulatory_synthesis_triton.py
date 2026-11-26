"""Triton 3.9 GPU kernel for fused central neuromodulatory convergence and multi-transmitter synthesis.

Synthesizes 5 ascending neuromodulatory axes (ACh, NE, 5-HT, DA, LHb) into integrated
cortical gain, net dopamine consolidation, waiting capacity, and motor readiness.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_neuromodulatory_synthesis_kernel(
    ach_ptr,
    ne_ptr,
    sht_ptr,
    da_ptr,
    lhb_ptr,
    out_gain_ptr,
    out_net_da_ptr,
    out_cap_ptr,
    out_ready_ptr,
    c_ach,
    c_ne,
    c_sht,
    gain_max,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load 5-axis transmitter vectors
    ach = tl.load(ach_ptr + offsets, mask=mask, other=0.0)
    ne = tl.load(ne_ptr + offsets, mask=mask, other=0.0)
    sht = tl.load(sht_ptr + offsets, mask=mask, other=0.0)
    da = tl.load(da_ptr + offsets, mask=mask, other=0.0)
    lhb = tl.load(lhb_ptr + offsets, mask=mask, other=0.0)

    # 1. Effective Cortical Gain: G = min(gain_max, 1.0 + c_ach * ACh + c_ne * NE)
    gain_raw = 1.0 + c_ach * ach + c_ne * ne
    gain = tl.minimum(gain_max, gain_raw)

    # 2. Net Dopaminergic Signal: DA_net = max(0.0, DA - LHb)
    da_net = tl.maximum(0.0, da - lhb)

    # 3. Waiting Capacity: P = 1.0 + c_sht * 5HT
    patience = 1.0 + c_sht * sht

    # 4. Motor Execution Readiness: R = (G * DA_net) * 0.10
    ready = gain * da_net * 0.10

    # Store results
    tl.store(out_gain_ptr + offsets, gain, mask=mask)
    tl.store(out_net_da_ptr + offsets, da_net, mask=mask)
    tl.store(out_cap_ptr + offsets, patience, mask=mask)
    tl.store(out_ready_ptr + offsets, ready, mask=mask)


def launch_fused_neuromodulatory_synthesis(
    ach_tone: torch.Tensor,
    ne_rate: torch.Tensor,
    serotonin_tone: torch.Tensor,
    da_rpe: torch.Tensor,
    lhb_stress: torch.Tensor,
    c_ach: float = 0.35,
    c_ne: float = 0.35,
    c_sht: float = 0.45,
    gain_max: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused neuromodulatory synthesis kernel."""
    assert ach_tone.is_contiguous()
    assert ne_rate.is_contiguous()
    assert serotonin_tone.is_contiguous()
    assert da_rpe.is_contiguous()
    assert lhb_stress.is_contiguous()

    n_elements = ach_tone.numel()
    out_gain = torch.empty_like(ach_tone)
    out_net_da = torch.empty_like(ach_tone)
    out_cap = torch.empty_like(ach_tone)
    out_ready = torch.empty_like(ach_tone)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_neuromodulatory_synthesis_kernel[grid](
        ach_tone,
        ne_rate,
        serotonin_tone,
        da_rpe,
        lhb_stress,
        out_gain,
        out_net_da,
        out_cap,
        out_ready,
        c_ach,
        c_ne,
        c_sht,
        gain_max,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_gain, out_net_da, out_cap, out_ready
