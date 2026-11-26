"""
Fused Triton 3.9 Kernel: Milestone Target #400
Huntington's Disease: Striatal D2-SPN Degeneration & Hyperkinetic Chorea
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _huntington_chorea_kernel(
    d2_surv_ptr,
    gaba_rescue_ptr,
    gpe_rate_ptr,
    stn_rate_ptr,
    thalamus_drive_ptr,
    chorea_score_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    surv = tl.load(d2_surv_ptr + offsets, mask=mask, other=1.0)
    resc = tl.load(gaba_rescue_ptr + offsets, mask=mask, other=0.0)

    surv = tl.maximum(0.0, tl.minimum(1.0, surv))
    resc = tl.maximum(0.0, tl.minimum(1.0, resc))

    d2_loss = 1.0 - surv
    resc_factor = 1.0 - resc * 0.75

    # GPe firing rate (Hz)
    gpe = 52.0 + 48.0 * d2_loss * resc_factor
    tl.store(gpe_rate_ptr + offsets, gpe, mask=mask)

    # STN motor brake rate (Hz)
    gpe_excess = tl.maximum(0.0, gpe - 50.0)
    stn = tl.maximum(2.0, 24.0 - 0.40 * gpe_excess)
    tl.store(stn_rate_ptr + offsets, stn, mask=mask)

    # Thalamocortical motor drive (%)
    stn_deficit = tl.maximum(0.0, 25.0 - stn)
    thal = tl.minimum(100.0, 20.0 + 3.8 * stn_deficit)
    tl.store(thalamus_drive_ptr + offsets, thal, mask=mask)

    # Chorea score (0 to 10)
    thal_excess = tl.maximum(0.0, thal - 40.0)
    chorea = tl.minimum(10.0, thal_excess / 5.5)
    tl.store(chorea_score_ptr + offsets, chorea, mask=mask)


def compute_huntington_chorea_triton(d2_survival: torch.Tensor, gaba_rescue: torch.Tensor):
    """
    Python wrapper executing fused Triton Huntington chorea kernel.
    """
    assert d2_survival.is_cuda == gaba_rescue.is_cuda
    n_elements = d2_survival.numel()
    device = d2_survival.device

    gpe_rate = torch.empty(n_elements, device=device, dtype=torch.float32)
    stn_rate = torch.empty(n_elements, device=device, dtype=torch.float32)
    thalamus_drive = torch.empty(n_elements, device=device, dtype=torch.float32)
    chorea_score = torch.empty(n_elements, device=device, dtype=torch.float32)

    if not d2_survival.is_cuda:
        # Fallback CPU execution
        surv = torch.clamp(d2_survival, 0.0, 1.0)
        resc = torch.clamp(gaba_rescue, 0.0, 1.0)
        d2_loss = 1.0 - surv
        resc_factor = 1.0 - resc * 0.75
        gpe = 52.0 + 48.0 * d2_loss * resc_factor
        gpe_excess = torch.clamp(gpe - 50.0, min=0.0)
        stn = torch.clamp(24.0 - 0.40 * gpe_excess, min=2.0)
        stn_deficit = torch.clamp(25.0 - stn, min=0.0)
        thal = torch.clamp(20.0 + 3.8 * stn_deficit, max=100.0)
        thal_excess = torch.clamp(thal - 40.0, min=0.0)
        chorea = torch.clamp(thal_excess / 5.5, max=10.0)
        return gpe, stn, thal, chorea

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _huntington_chorea_kernel[grid](
        d2_survival,
        gaba_rescue,
        gpe_rate,
        stn_rate,
        thalamus_drive,
        chorea_score,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return gpe_rate, stn_rate, thalamus_drive, chorea_score
