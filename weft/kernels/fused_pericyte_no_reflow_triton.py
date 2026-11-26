"""
Fused Triton Kernel for Pericyte Capillary Constriction & Ischemic No-Reflow Dynamics.
Target #412
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def pericyte_no_reflow_kernel(
    ca_ptr,
    baseline_d_ptr,
    shed_rescue_ptr,
    out_diam_ptr,
    out_flux_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ca = tl.load(ca_ptr + offsets, mask=mask, other=120.0)
    d0 = tl.load(baseline_d_ptr + offsets, mask=mask, other=5.5)
    rescue = tl.load(shed_rescue_ptr + offsets, mask=mask, other=0.0)

    # Pericyte contraction delta
    delta_ca = ca * 0.0075
    delta_d = (rescue * 0.02) - delta_ca
    d_new = d0 + delta_d

    # Clamp min lumen to 2.0 um
    d_clamped = tl.maximum(d_new, 2.0)

    # Poiseuille flux ratio (d / d0)^4
    ratio = d_clamped / d0
    ratio2 = ratio * ratio
    flux = ratio2 * ratio2

    tl.store(out_diam_ptr + offsets, d_clamped, mask=mask)
    tl.store(out_flux_ptr + offsets, flux, mask=mask)

def run_fused_pericyte_no_reflow(ca, baseline_d, shed_rescue):
    n_elements = ca.numel()
    out_diam = torch.empty_like(baseline_d)
    out_flux = torch.empty_like(baseline_d)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    pericyte_no_reflow_kernel[grid](
        ca, baseline_d, shed_rescue, out_diam, out_flux, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_diam, out_flux
