"""
Fused Triton Kernel for OPC Differentiation & Axon Conduction Dynamics.
Target #414
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def opc_conduction_kernel(
    insult_ptr,
    shed_ptr,
    out_mbp_ptr,
    out_vel_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    insult = tl.load(insult_ptr + offsets, mask=mask, other=0.0)
    shed = tl.load(shed_ptr + offsets, mask=mask, other=0.0)

    is_demyel = insult > 50.0

    mbp_demyel = 25.0 + shed * 1.25
    mbp_norm = 150.0 - insult * 0.3
    mbp = tl.where(is_demyel, mbp_demyel, mbp_norm)

    vel_demyel = 15.0 + shed * 0.6
    vel_norm = 80.0 - insult * 0.2
    vel = tl.where(is_demyel, vel_demyel, vel_norm)

    tl.store(out_mbp_ptr + offsets, mbp, mask=mask)
    tl.store(out_vel_ptr + offsets, vel, mask=mask)

def run_fused_opc_conduction(insult, shed):
    n_elements = insult.numel()
    out_mbp = torch.empty_like(insult)
    out_vel = torch.empty_like(insult)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    opc_conduction_kernel[grid](
        insult, shed, out_mbp, out_vel, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_mbp, out_vel
