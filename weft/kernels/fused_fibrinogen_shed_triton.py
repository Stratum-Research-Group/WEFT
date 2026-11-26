"""
Fused Triton 3.9 Kernel: Target #406 Fibrinogen-SHED-CM Molecular Interactions
Block-level parallel simulation of proteomic chaperone shielding, thrombin cleavage inhibition,
and microglial Mac-1 neuroinflammatory suppression.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_fibrinogen_shed_kernel(
    fib_ptr,
    thr_ptr,
    abeta_ptr,
    shed_dose_ptr,
    clot_out_ptr,
    mac1_out_ptr,
    spine_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    fib = tl.load(fib_ptr + offsets, mask=mask, other=0.0)
    thr = tl.load(thr_ptr + offsets, mask=mask, other=0.0)
    abeta = tl.load(abeta_ptr + offsets, mask=mask, other=0.0)
    dose = tl.load(shed_dose_ptr + offsets, mask=mask, other=0.0)

    # Raw clot drive = (fib / 3.0) * (thr / 1.5) * (1.0 + abeta / 50.0) * 35.0
    f_part = fib / 3.0
    t_part = thr / 1.5
    a_part = 1.0 + abeta / 50.0
    raw_clot = tl.minimum(100.0, f_part * t_part * a_part * 35.0)

    # Secretome shielding = min(0.92, (dose / 100.0) * 0.95)
    shielding = tl.minimum(0.92, (dose / 100.0) * 0.95)

    # Effective amyloidogenic fibrin clot
    eff_clot = raw_clot * (1.0 - shielding)
    tl.store(clot_out_ptr + offsets, eff_clot, mask=mask)

    # Mac-1 microglial activation score = eff_clot * 0.92 * (1.0 - (dose / 100.0) * 0.35)
    mac1 = eff_clot * 0.92 * (1.0 - (dose / 100.0) * 0.35)
    tl.store(mac1_out_ptr + offsets, mac1, mask=mask)

    # Synaptic spine loss index = mac1 * 0.82
    spine_loss = mac1 * 0.82
    tl.store(spine_out_ptr + offsets, spine_loss, mask=mask)

def launch_fibrinogen_shed_triton(fib, thr, abeta, shed_dose):
    assert fib.is_cuda and thr.is_cuda and abeta.is_cuda and shed_dose.is_cuda
    n_elements = fib.numel()
    clot_out = torch.empty_like(fib)
    mac1_out = torch.empty_like(fib)
    spine_out = torch.empty_like(fib)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_fibrinogen_shed_kernel[grid](
        fib, thr, abeta, shed_dose,
        clot_out, mac1_out, spine_out,
        n_elements,
        BLOCK_SIZE=1024
    )
    return clot_out, mac1_out, spine_out
