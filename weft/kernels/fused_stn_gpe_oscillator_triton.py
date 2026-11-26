"""
Triton 3.9 Fused STN-GPe Closed-Loop Oscillator Kernel.
Calculates:
1. Prototypic vs Arkypallidal GPe firing rates.
2. Arkypallidal feedback cancellation of striatal motor output.
3. Pathological STN-GPe reciprocal beta resonance frequency synthesis.
Target #383 in WEFT Verification Suite.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_stn_gpe_oscillator_kernel(
    da_tone_ptr,
    striatal_drive_ptr,
    stop_active_ptr,
    arky_rate_out_ptr,
    striatal_out_ptr,
    beta_freq_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    da = tl.load(da_tone_ptr + offsets, mask=mask, other=50.0)
    drive = tl.load(striatal_drive_ptr + offsets, mask=mask, other=0.0)
    stop = tl.load(stop_active_ptr + offsets, mask=mask, other=0.0)

    # da_def = max(0, 100 - da)
    da_def = tl.maximum(0.0, 100.0 - da)

    # Arkypallidal rate: stop ? 55.0 : (8.0 + da_def * 0.08)
    base_arky = 8.0 + da_def * 0.08
    arky = tl.where(stop > 0.5, 55.0, base_arky)
    tl.store(arky_rate_out_ptr + offsets, arky, mask=mask)

    # Striatal output: max(0, drive - drive * arky / 60)
    inhib = (drive * arky) / 60.0
    suppressed = tl.maximum(0.0, drive - inhib)
    tl.store(striatal_out_ptr + offsets, suppressed, mask=mask)

    # Beta frequency: 13.0 + da_def * 0.17
    f_res = 13.0 + (da_def * 17.0) / 100.0
    tl.store(beta_freq_out_ptr + offsets, f_res, mask=mask)

def launch_stn_gpe_oscillator_transform(
    da_tone: torch.Tensor,
    striatal_drive: torch.Tensor,
    stop_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = da_tone.numel()
    arky_out = torch.empty_like(da_tone)
    striatal_out = torch.empty_like(striatal_drive)
    beta_freq_out = torch.empty_like(da_tone)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_stn_gpe_oscillator_kernel[grid](
        da_tone,
        striatal_drive,
        stop_active,
        arky_out,
        striatal_out,
        beta_freq_out,
        n,
        BLOCK_SIZE=256,
    )
    return arky_out, striatal_out, beta_freq_out
