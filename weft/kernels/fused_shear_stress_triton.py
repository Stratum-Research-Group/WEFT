"""
Fused Triton Kernel for Endothelial Shear-Stress Mechanotransduction & eNOS Phosphorylation.
Target #413
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def shear_stress_enos_kernel(
    shear_ptr,
    shed_ptr,
    out_phos_ptr,
    out_no_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tau = tl.load(shear_ptr + offsets, mask=mask, other=0.0)
    shed = tl.load(shed_ptr + offsets, mask=mask, other=0.0)

    # Mechanotransduction threshold: tau >= 12 dyn/cm2
    is_laminar = tau >= 12.0

    phos_lam = 70.0 + tau * 0.8
    phos_dist = 25.0 + shed * 0.45
    phos = tl.where(is_laminar, phos_lam, phos_dist)

    no_lam = 1.8 + tau * 0.04
    no_dist = 0.6 + shed * 0.015
    no_bio = tl.where(is_laminar, no_lam, no_dist)

    tl.store(out_phos_ptr + offsets, phos, mask=mask)
    tl.store(out_no_ptr + offsets, no_bio, mask=mask)

def run_fused_shear_stress_enos(shear, shed):
    n_elements = shear.numel()
    out_phos = torch.empty_like(shear)
    out_no = torch.empty_like(shear)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    shear_stress_enos_kernel[grid](
        shear, shed, out_phos, out_no, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_phos, out_no
