"""
Fused Triton 3.9 Kernel: Target #398
Striatal Cholinergic Interneuron (ChI) Hyper-Rebound & Dystonic Co-Contraction
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _striatal_cholinergic_kernel(
    d2_tone_ptr,
    m4_rescue_ptr,
    chi_rate_ptr,
    ach_conc_ptr,
    agonist_drive_ptr,
    antagonist_leak_ptr,
    cci_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    d2 = tl.load(d2_tone_ptr + offsets, mask=mask, other=1.0)
    m4 = tl.load(m4_rescue_ptr + offsets, mask=mask, other=0.0)

    # Clamping
    d2 = tl.maximum(0.0, tl.minimum(1.0, d2))
    m4 = tl.maximum(0.0, tl.minimum(1.0, m4))

    d2_loss = 1.0 - d2
    m4_suppr = 1.0 - m4 * 0.70

    # ChI firing rate (Hz)
    rate = 4.5 + 25.0 * d2_loss * m4_suppr
    tl.store(chi_rate_ptr + offsets, rate, mask=mask)

    # [ACh] (nM)
    ach = rate * 4.2
    tl.store(ach_conc_ptr + offsets, ach, mask=mask)

    # Agonist drive
    ag = 30.0 + 0.25 * rate
    tl.store(agonist_drive_ptr + offsets, ag, mask=mask)

    # Antagonist leakage
    ant_extra = rate * 0.20 * (1.0 - 0.85 * m4)
    ant = 8.0 + tl.where(d2 < 0.40, ant_extra, 0.0)
    tl.store(antagonist_leak_ptr + offsets, ant, mask=mask)

    # Co-Contraction Index (CCI) = 2.0 * min(ag, ant) / (ag + ant)
    min_act = tl.minimum(ag, ant)
    sum_act = ag + ant
    cci_val = (2.0 * min_act) / sum_act
    tl.store(cci_ptr + offsets, cci_val, mask=mask)


def compute_striatal_cholinergic_triton(d2_tone: torch.Tensor, m4_rescue: torch.Tensor):
    """
    Python wrapper executing fused Triton kernel on GPU/CPU tensors.
    """
    assert d2_tone.is_cuda == m4_rescue.is_cuda
    n_elements = d2_tone.numel()
    device = d2_tone.device

    chi_rate = torch.empty(n_elements, device=device, dtype=torch.float32)
    ach_conc = torch.empty(n_elements, device=device, dtype=torch.float32)
    agonist_drive = torch.empty(n_elements, device=device, dtype=torch.float32)
    antagonist_leak = torch.empty(n_elements, device=device, dtype=torch.float32)
    cci = torch.empty(n_elements, device=device, dtype=torch.float32)

    if not d2_tone.is_cuda:
        # Fallback reference vector execution on CPU
        d2 = torch.clamp(d2_tone, 0.0, 1.0)
        m4 = torch.clamp(m4_rescue, 0.0, 1.0)
        rate = 4.5 + 25.0 * (1.0 - d2) * (1.0 - m4 * 0.70)
        ach = rate * 4.2
        ag = 30.0 + 0.25 * rate
        ant_extra = rate * 0.20 * (1.0 - 0.85 * m4)
        ant = 8.0 + torch.where(d2 < 0.40, ant_extra, torch.zeros_like(rate))
        cci_val = (2.0 * torch.minimum(ag, ant)) / (ag + ant)
        return rate, ach, ag, ant, cci_val

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _striatal_cholinergic_kernel[grid](
        d2_tone,
        m4_rescue,
        chi_rate,
        ach_conc,
        agonist_drive,
        antagonist_leak,
        cci,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return chi_rate, ach_conc, agonist_drive, antagonist_leak, cci
