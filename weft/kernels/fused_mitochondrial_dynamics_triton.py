"""
Fused Triton Kernel for Presynaptic Mitochondrial Dynamics & Synaptosomal ATP.
Target #415
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def mitochondrial_dynamics_kernel(
    stress_ptr,
    shed_ptr,
    out_psi_ptr,
    out_atp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    stress = tl.load(stress_ptr + offsets, mask=mask, other=0.0)
    shed = tl.load(shed_ptr + offsets, mask=mask, other=0.0)

    is_stress = stress > 50.0

    psi_stress = 100.0 + shed * 0.55
    psi_norm = 165.0 - stress * 0.2
    psi = tl.where(is_stress, psi_stress, psi_norm)

    atp_stress = 1.5 + shed * 0.02
    atp_norm = 3.8 - stress * 0.01
    atp = tl.where(is_stress, atp_stress, atp_norm)

    tl.store(out_psi_ptr + offsets, psi, mask=mask)
    tl.store(out_atp_ptr + offsets, atp, mask=mask)

def run_fused_mitochondrial_dynamics(stress, shed):
    n_elements = stress.numel()
    out_psi = torch.empty_like(stress)
    out_atp = torch.empty_like(stress)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    mitochondrial_dynamics_kernel[grid](
        stress, shed, out_psi, out_atp, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_psi, out_atp
