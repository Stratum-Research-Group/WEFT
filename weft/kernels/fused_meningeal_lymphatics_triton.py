"""
Fused Triton Kernel for Dural Meningeal Lymphatics & Deep Cervical Clearance.
Target #417
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def meningeal_lymphatics_kernel(
    aging_ptr,
    vegfc_ptr,
    out_vel_ptr,
    out_eff_ptr,
    out_clr_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    age = tl.load(aging_ptr + offsets, mask=mask, other=0.0)
    vegfc = tl.load(vegfc_ptr + offsets, mask=mask, other=0.0)

    is_aged = age > 50.0

    vel_aged = 1.5 + vegfc * 0.035
    vel_norm = 5.5 - age * 0.015
    vel = tl.where(is_aged, vel_aged, vel_norm)

    eff_aged = 0.8 + vegfc * 0.022
    eff_norm = 3.2 - age * 0.008
    eff = tl.where(is_aged, eff_aged, eff_norm)

    clr_aged = 20.0 + vegfc * 0.55
    clr_norm = 80.0 - age * 0.2
    clr = tl.where(is_aged, clr_aged, clr_norm)

    tl.store(out_vel_ptr + offsets, vel, mask=mask)
    tl.store(out_eff_ptr + offsets, eff, mask=mask)
    tl.store(out_clr_ptr + offsets, clr, mask=mask)

def run_fused_meningeal_lymphatics(aging, vegfc):
    n_elements = aging.numel()
    out_vel = torch.empty_like(aging)
    out_eff = torch.empty_like(aging)
    out_clr = torch.empty_like(aging)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    meningeal_lymphatics_kernel[grid](
        aging, vegfc, out_vel, out_eff, out_clr, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_vel, out_eff, out_clr
