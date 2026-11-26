"""
Fused Triton 3.9 Kernel: Target #404 Paroxysmal Dyskinesia & Channelopathy
Block-level parallel simulation of cerebellar-thalamo-striatal disynaptic loop & carbamazepine rescue.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_paroxysmal_dyskinesia_kernel(
    prrt2_ptr,
    trigger_ptr,
    cbz_ptr,
    burst_out_ptr,
    da_out_ptr,
    sev_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    prrt2 = tl.load(prrt2_ptr + offsets, mask=mask, other=1.0)
    trigger = tl.load(trigger_ptr + offsets, mask=mask, other=0.0)
    cbz = tl.load(cbz_ptr + offsets, mask=mask, other=0.0)

    # Deficit = max(0.0, 1.0 - PRRT2)
    deficit = tl.maximum(0.0, 1.0 - prrt2)

    # Raw DCN burst = 40.0 + deficit * 120.0 * trigger
    raw_burst = 40.0 + deficit * 120.0 * trigger

    # Nav channel block = min(1.0, cbz / 300.0)
    nav_inh = tl.minimum(1.0, cbz / 300.0)

    # Rescued burst = raw_burst * (1.0 - 0.75 * nav_inh)
    rescued_burst = raw_burst * (1.0 - 0.75 * nav_inh)
    tl.store(burst_out_ptr + offsets, rescued_burst, mask=mask)

    # Striatal DA efflux = 1.0 + (burst / 60.0) * 1.2
    da_efflux = 1.0 + (rescued_burst / 60.0) * 1.2
    tl.store(da_out_ptr + offsets, da_efflux, mask=mask)

    # Dyskinesia severity = clamp(0.0, 100.0, (burst - 45.0)*0.95 + (da - 1.2)*22.0)
    b_excess = (rescued_burst - 45.0) * 0.95
    da_excess = (da_efflux - 1.2) * 22.0
    raw_sev = b_excess + da_excess
    severity = tl.maximum(0.0, tl.minimum(100.0, raw_sev))
    tl.store(sev_out_ptr + offsets, severity, mask=mask)

def launch_paroxysmal_dyskinesia_triton(prrt2, trigger, cbz):
    assert prrt2.is_cuda and trigger.is_cuda and cbz.is_cuda
    n_elements = prrt2.numel()
    burst_out = torch.empty_like(prrt2)
    da_out = torch.empty_like(prrt2)
    sev_out = torch.empty_like(prrt2)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_paroxysmal_dyskinesia_kernel[grid](
        prrt2, trigger, cbz,
        burst_out, da_out, sev_out,
        n_elements,
        BLOCK_SIZE=1024
    )
    return burst_out, da_out, sev_out
