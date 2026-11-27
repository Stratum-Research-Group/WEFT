"""
Fused Triton 3.9 Kernel: Hypothalamic AgRP/POMC Metabolic Energy Homeostasis (Target #427).
Optimized for parallel execution across arcuate nucleus neural arrays.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def agrp_pomc_kernel(
    ghrelin_ptr,
    leptin_ptr,
    agrp_out_ptr,
    pomc_out_ptr,
    mc4r_out_ptr,
    ee_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    ghrelin = tl.load(ghrelin_ptr + offsets, mask=mask, other=0.0)
    leptin = tl.load(leptin_ptr + offsets, mask=mask, other=0.0)

    # AgRP firing rate calculation
    agrp_raw = ghrelin * 0.38 - leptin * 0.15 + 5.0
    agrp = tl.maximum(agrp_raw, 1.0)

    # POMC firing rate calculation
    pomc_raw = leptin * 0.28 - agrp * 0.35 + 3.0
    pomc = tl.maximum(pomc_raw, 0.5)

    # MC4R signaling level
    mc4r_raw = pomc * 4.2 - agrp * 1.8 + 30.0
    mc4r = tl.maximum(5.0, tl.minimum(mc4r_raw, 95.0))

    # Autonomic energy expenditure
    ee = mc4r * 0.85 + 5.0

    # Store outputs
    tl.store(agrp_out_ptr + offsets, agrp, mask=mask)
    tl.store(pomc_out_ptr + offsets, pomc, mask=mask)
    tl.store(mc4r_out_ptr + offsets, mc4r, mask=mask)
    tl.store(ee_out_ptr + offsets, ee, mask=mask)

def launch_agrp_pomc(ghrelin: torch.Tensor, leptin: torch.Tensor):
    assert ghrelin.is_cuda and leptin.is_cuda, "Inputs must be CUDA tensors"
    assert ghrelin.shape == leptin.shape, "Input shapes must match"

    n_elements = ghrelin.numel()
    agrp_out = torch.empty_like(ghrelin)
    pomc_out = torch.empty_like(ghrelin)
    mc4r_out = torch.empty_like(ghrelin)
    ee_out = torch.empty_like(ghrelin)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    agrp_pomc_kernel[grid](
        ghrelin,
        leptin,
        agrp_out,
        pomc_out,
        mc4r_out,
        ee_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return agrp_out, pomc_out, mc4r_out, ee_out
