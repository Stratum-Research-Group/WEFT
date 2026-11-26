"""
Fused Triton 3.9 Kernel: Target #410 Glymphatic Convective Flux & AQP4 Polarization
Block-level parallel simulation of astrocytic endfoot AQP4 water transport,
NREM slow-wave sleep interstitial volume expansion (alpha), and convective solute clearance.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_glymphatic_aqp4_kernel(
    pol_ptr,
    delta_ptr,
    vaso_ptr,
    shed_dose_ptr,
    flux_out_ptr,
    clearance_out_ptr,
    retention_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(pol_ptr + offsets, mask=mask, other=35.0)
    d = tl.load(delta_ptr + offsets, mask=mask, other=0.5)
    v = tl.load(vaso_ptr + offsets, mask=mask, other=2.0)
    s = tl.load(shed_dose_ptr + offsets, mask=mask, other=0.0)

    # Effective AQP4 polarization = min(100.0, max(15.0, p + s * 0.55))
    eff_pol = tl.minimum(100.0, tl.maximum(15.0, p + s * 0.55))

    # Interstitial volume fraction alpha = 14.0 + (d * 10.0) * (1.0 + s * 0.20 / 100)
    dose_mod = 1.0 + s * (0.20 / 100.0)
    eff_alpha = tl.minimum(25.0, tl.maximum(12.0, 14.0 + (d * 10.0) * dose_mod))

    # Convective flux (um/min) = (v / 3.5) * (eff_pol / 85.0) * (eff_alpha / 22.0) * 2.1
    v_norm = v / 3.5
    p_norm = eff_pol / 85.0
    a_norm = eff_alpha / 22.0
    flux = tl.maximum(0.1, tl.minimum(3.5, v_norm * p_norm * a_norm * 2.1))
    tl.store(flux_out_ptr + offsets, flux, mask=mask)

    # Clearance (%/hr) = min(100.0, max(5.0, (flux / 2.0) * 75.0))
    clearance = tl.minimum(100.0, tl.maximum(5.0, (flux / 2.0) * 75.0))
    tl.store(clearance_out_ptr + offsets, clearance, mask=mask)

    # Retention = 100.0 - clearance
    retention = tl.maximum(0.0, 100.0 - clearance)
    tl.store(retention_out_ptr + offsets, retention, mask=mask)

def launch_glymphatic_aqp4_triton(pol, delta, vaso, shed_dose):
    assert pol.is_cuda and delta.is_cuda and vaso.is_cuda and shed_dose.is_cuda
    n_elements = pol.numel()
    flux_out = torch.empty_like(pol)
    clearance_out = torch.empty_like(pol)
    retention_out = torch.empty_like(pol)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_glymphatic_aqp4_kernel[grid](
        pol, delta, vaso, shed_dose,
        flux_out, clearance_out, retention_out,
        n_elements,
        BLOCK_SIZE=1024
    )
    return flux_out, clearance_out, retention_out
