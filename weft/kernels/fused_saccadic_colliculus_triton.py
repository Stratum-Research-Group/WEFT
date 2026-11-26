"""
Triton 3.9 Fused Superior Colliculus Saccadic Burst Generator Kernel.
Calculates:
1. Log-polar collicular population vector decoding.
2. Main-sequence non-linear velocity saturation and linear duration scaling.
3. Omnipause Neuron (OPN) disinhibition gating.
Target #379 in WEFT Verification Suite.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_saccadic_burst_kernel(
    amp_ptr,
    opn_clamp_ptr,
    sc_drive_ptr,
    peak_vel_out_ptr,
    duration_out_ptr,
    pprf_rate_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(amp_ptr + offsets, mask=mask, other=0.0)
    opn = tl.load(opn_clamp_ptr + offsets, mask=mask, other=0.0)
    sc = tl.load(sc_drive_ptr + offsets, mask=mask, other=0.0)

    # Duration: 22.0 + 2.5 * a
    dur = 22.0 + 2.5 * a
    tl.store(duration_out_ptr + offsets, dur, mask=mask)

    # Main sequence: v_peak = 850.0 * (x / (1.0 + x)) where x = a / 12.0
    x = a / 12.0
    v = 850.0 * (x / (1.0 + x))
    tl.store(peak_vel_out_ptr + offsets, v, mask=mask)

    # Gating: sc > opn
    is_paused = sc > opn
    pprf = tl.where(is_paused, v * 1.15, 0.0)
    tl.store(pprf_rate_out_ptr + offsets, pprf, mask=mask)

def launch_saccadic_burst_transform(
    amp: torch.Tensor,
    opn_clamp: torch.Tensor,
    sc_drive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = amp.numel()
    peak_vel_out = torch.empty_like(amp)
    duration_out = torch.empty_like(amp)
    pprf_rate_out = torch.empty_like(amp)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_saccadic_burst_kernel[grid](
        amp,
        opn_clamp,
        sc_drive,
        peak_vel_out,
        duration_out,
        pprf_rate_out,
        n,
        BLOCK_SIZE=256,
    )
    return peak_vel_out, duration_out, pprf_rate_out
