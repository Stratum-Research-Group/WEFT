"""
Fused Triton Kernel for Neurovascular Coupling & BK_Ca Vasomotor Dynamics.
Target #411
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def neurovascular_coupling_kernel(
    ca_ptr,
    radius_ptr,
    eet_ptr,
    out_radius_ptr,
    out_flux_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ca = tl.load(ca_ptr + offsets, mask=mask, other=0.0)
    r0 = tl.load(radius_ptr + offsets, mask=mask, other=20.0)
    eet = tl.load(eet_ptr + offsets, mask=mask, other=0.0)

    # Vasomotor gating: Ca < 500 nM triggers BK_Ca dilation; Ca >= 500 triggers paradoxical constriction
    is_dilatory = ca < 500.0
    delta_ca = tl.where(is_dilatory, ca * 0.025, -ca * 0.015)
    delta_r = delta_ca + eet * 0.05
    r_new = r0 + delta_r

    # Poiseuille Q ~ (r / r0)^4
    ratio = r_new / r0
    ratio2 = ratio * ratio
    flux = ratio2 * ratio2

    tl.store(out_radius_ptr + offsets, r_new, mask=mask)
    tl.store(out_flux_ptr + offsets, flux, mask=mask)

def run_fused_neurovascular_coupling(ca, radius, eet):
    n_elements = ca.numel()
    out_radius = torch.empty_like(radius)
    out_flux = torch.empty_like(radius)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    neurovascular_coupling_kernel[grid](
        ca, radius, eet, out_radius, out_flux, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_radius, out_flux
