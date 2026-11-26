"""
Fused Triton 3.9 Kernel: Target #407 Blood-Brain Barrier Tight-Junction Restoration
Block-level parallel simulation of pericyte-endothelial Tie2 phosphorylation,
Claudin-5/Occludin reconstitution, and trans-endothelial electrical resistance (TEER).
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_bbb_tight_junction_kernel(
    pericyte_ptr,
    ang1_ptr,
    ros_ptr,
    mmp_ptr,
    shed_dose_ptr,
    tie2_out_ptr,
    claudin_out_ptr,
    teer_out_ptr,
    leak_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(pericyte_ptr + offsets, mask=mask, other=85.0)
    a = tl.load(ang1_ptr + offsets, mask=mask, other=18.0)
    r = tl.load(ros_ptr + offsets, mask=mask, other=20.0)
    m = tl.load(mmp_ptr + offsets, mask=mask, other=1.5)
    d = tl.load(shed_dose_ptr + offsets, mask=mask, other=100.0)

    # Tie2 phosphorylation = min(100.0, max(10.0, (20.0 + p*0.3 + a*3.2) * (1.0 - r/200.0)))
    raw_t = 20.0 + p * 0.3 + a * 3.2
    ros_factor = 1.0 - (r / 200.0)
    eff_t = tl.minimum(100.0, tl.maximum(10.0, raw_t * ros_factor))
    tl.store(tie2_out_ptr + offsets, eff_t, mask=mask)

    # MMP suppression by TIMP-1/2 in secretome = max(0.2, 1.0 - d * 0.75/100)
    m_supp = tl.maximum(0.2, 1.0 - d * (0.75 / 100.0))
    eff_m = m * m_supp

    # Claudin-5 and Occludin expression
    claudin = tl.minimum(100.0, tl.maximum(5.0, eff_t * 1.1 - eff_m * 12.0))
    tl.store(claudin_out_ptr + offsets, claudin, mask=mask)

    occludin = tl.minimum(100.0, tl.maximum(5.0, eff_t * 1.05 - eff_m * 10.5))

    # Trans-endothelial electrical resistance (TEER)
    teer = 200.0 + claudin * 8.5 + occludin * 4.5
    tl.store(teer_out_ptr + offsets, teer, mask=mask)

    # Paracellular leakage
    leakage = tl.maximum(2.0, tl.minimum(100.0, 100.0 - teer * (100.0 / 1500.0)))
    tl.store(leak_out_ptr + offsets, leakage, mask=mask)

def launch_bbb_tight_junction_triton(pericyte, ang1, ros, mmp, shed_dose):
    assert pericyte.is_cuda and ang1.is_cuda and ros.is_cuda and mmp.is_cuda and shed_dose.is_cuda
    n_elements = pericyte.numel()
    tie2_out = torch.empty_like(pericyte)
    claudin_out = torch.empty_like(pericyte)
    teer_out = torch.empty_like(pericyte)
    leak_out = torch.empty_like(pericyte)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bbb_tight_junction_kernel[grid](
        pericyte, ang1, ros, mmp, shed_dose,
        tie2_out, claudin_out, teer_out, leak_out,
        n_elements,
        BLOCK_SIZE=1024
    )
    return tie2_out, claudin_out, teer_out, leak_out
