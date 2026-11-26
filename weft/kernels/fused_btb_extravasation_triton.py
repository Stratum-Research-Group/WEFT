"""
Fused Triton Kernel for Blood-Tumor Barrier (BTB) Heterogeneity & Convective Extravasation.
Target #416
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def btb_extravasation_kernel(
    stage_ptr,
    rescue_ptr,
    out_ifp_ptr,
    out_flux_ptr,
    out_delivery_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    stage = tl.load(stage_ptr + offsets, mask=mask, other=0.0)
    rescue = tl.load(rescue_ptr + offsets, mask=mask, other=0.0)

    is_tumor = stage > 50.0

    ifp_tum = 28.0 - rescue * 0.22
    ifp_nor = 4.0 + stage * 0.1
    ifp = tl.where(is_tumor, ifp_tum, ifp_nor)

    flux_tum = 1.5 + rescue * 0.055
    flux_nor = 7.0 - stage * 0.02
    flux = tl.where(is_tumor, flux_tum, flux_nor)

    del_tum = 12.0 + rescue * 0.52
    del_nor = 65.0 - stage * 0.15
    delivery = tl.where(is_tumor, del_tum, del_nor)

    tl.store(out_ifp_ptr + offsets, ifp, mask=mask)
    tl.store(out_flux_ptr + offsets, flux, mask=mask)
    tl.store(out_delivery_ptr + offsets, delivery, mask=mask)

def run_fused_btb_extravasation(stage, rescue):
    n_elements = stage.numel()
    out_ifp = torch.empty_like(stage)
    out_flux = torch.empty_like(stage)
    out_delivery = torch.empty_like(stage)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    btb_extravasation_kernel[grid](
        stage, rescue, out_ifp, out_flux, out_delivery, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_ifp, out_flux, out_delivery
