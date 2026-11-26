"""
Fused Triton 3.9 Kernel: Target #402
Essential Tremor (ET): Inferior Olivary Pacemaking & DRT Hyper-Oscillation
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _essential_tremor_kernel(
    gap_coupling_ptr,
    cav3_cond_ptr,
    cav3_block_ptr,
    vim_dbs_ptr,
    ion_freq_ptr,
    drt_coherence_ptr,
    thal_power_ptr,
    tremor_amp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gap = tl.load(gap_coupling_ptr + offsets, mask=mask, other=0.85)
    cav = tl.load(cav3_cond_ptr + offsets, mask=mask, other=0.80)
    blk = tl.load(cav3_block_ptr + offsets, mask=mask, other=0.0)
    dbs = tl.load(vim_dbs_ptr + offsets, mask=mask, other=0.0)

    gap = tl.maximum(0.0, tl.minimum(1.0, gap))
    cav = tl.maximum(0.0, tl.minimum(1.0, cav))
    blk = tl.maximum(0.0, tl.minimum(1.0, blk))

    eff_cav = cav * (1.0 - blk * 0.85)
    eff_gap = gap * (1.0 - blk * 0.60)

    # Frequency
    freq = 4.2 + 2.5 * eff_cav
    tl.store(ion_freq_ptr + offsets, freq, mask=mask)

    # DRT coherence
    coh_mult = tl.where(eff_cav > 0.20, 0.90, 0.20)
    raw_coh = eff_gap * coh_mult

    dbs_atten = tl.where(dbs > 0.5, 0.15, 1.0)
    final_coh = tl.minimum(1.0, raw_coh * dbs_atten)
    tl.store(drt_coherence_ptr + offsets, final_coh, mask=mask)

    # Thalamic power
    raw_pow = eff_cav * 40.0 + eff_gap * 45.0
    pow = tl.maximum(5.0, raw_pow * dbs_atten)
    tl.store(thal_power_ptr + offsets, pow, mask=mask)

    # Tremor amplitude
    amp = tl.maximum(0.1, final_coh * 3.8)
    tl.store(tremor_amp_ptr + offsets, amp, mask=mask)


def compute_essential_tremor_triton(
    gap_coupling: torch.Tensor,
    cav3_cond: torch.Tensor,
    cav3_block: torch.Tensor,
    vim_dbs: torch.Tensor,
):
    """
    Python wrapper executing fused Triton Essential Tremor kernel.
    """
    assert gap_coupling.is_cuda == cav3_cond.is_cuda == cav3_block.is_cuda == vim_dbs.is_cuda
    n_elements = gap_coupling.numel()
    device = gap_coupling.device

    ion_freq = torch.empty(n_elements, device=device, dtype=torch.float32)
    drt_coherence = torch.empty(n_elements, device=device, dtype=torch.float32)
    thal_power = torch.empty(n_elements, device=device, dtype=torch.float32)
    tremor_amp = torch.empty(n_elements, device=device, dtype=torch.float32)

    if not gap_coupling.is_cuda:
        # Fallback CPU execution
        gap = torch.clamp(gap_coupling, 0.0, 1.0)
        cav = torch.clamp(cav3_cond, 0.0, 1.0)
        blk = torch.clamp(cav3_block, 0.0, 1.0)

        eff_cav = cav * (1.0 - blk * 0.85)
        eff_gap = gap * (1.0 - blk * 0.60)
        freq = 4.2 + 2.5 * eff_cav

        coh_mult = torch.where(eff_cav > 0.20, 0.90, 0.20)
        raw_coh = eff_gap * coh_mult
        dbs_atten = torch.where(vim_dbs > 0.5, 0.15, 1.0)
        final_coh = torch.clamp(raw_coh * dbs_atten, max=1.0)
        raw_pow = eff_cav * 40.0 + eff_gap * 45.0
        pow_val = torch.clamp(raw_pow * dbs_atten, min=5.0)
        amp = torch.clamp(final_coh * 3.8, min=0.1)
        return freq, final_coh, pow_val, amp

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _essential_tremor_kernel[grid](
        gap_coupling,
        cav3_cond,
        cav3_block,
        vim_dbs,
        ion_freq,
        drt_coherence,
        thal_power,
        tremor_amp,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return ion_freq, drt_coherence, thal_power, tremor_amp
