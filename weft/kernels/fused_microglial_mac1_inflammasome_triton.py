"""
Fused Triton 3.9 Kernel: Target #409 Microglial Mac-1 & NLRP3 Inflammasome Gating
Block-level parallel simulation of microglial Mac-1 integrin ligation, ASC speck assembly,
Caspase-1 cleavage, and M1/M2 phenotype polarization under SHED secretome infusion.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_microglial_mac1_inflammasome_kernel(
    gamma_ptr,
    tlr4_ptr,
    p2x7_ptr,
    shed_dose_ptr,
    mac1_out_ptr,
    speck_out_ptr,
    il1b_out_ptr,
    ratio_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    g = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(tlr4_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(p2x7_ptr + offsets, mask=mask, other=0.0)
    d = tl.load(shed_dose_ptr + offsets, mask=mask, other=0.0)

    # HGF masking = min(0.92, d * 0.95 / 100)
    shield = tl.minimum(0.92, d * (0.95 / 100.0))
    eff_gamma = g * (1.0 - shield)

    # Mac-1 ligation = max(2.0, min(100.0, eff_gamma * 1.05 * (1.0 - d * 0.65 / 100)))
    itgam_mod = 1.0 - (d * (0.65 / 100.0))
    raw_mac1 = eff_gamma * 1.05 * itgam_mod
    mac1 = tl.maximum(2.0, tl.minimum(100.0, raw_mac1))
    tl.store(mac1_out_ptr + offsets, mac1, mask=mask)

    # Priming and activation
    raw_prime = t * 0.40 + mac1 * 0.60
    prime = raw_prime * (1.0 - (d * (0.70 / 100.0)))
    act = p * (1.0 - (d * (0.50 / 100.0)))

    # ASC speck assembly
    speck = tl.minimum(100.0, (prime * act) / 75.0)
    tl.store(speck_out_ptr + offsets, speck, mask=mask)

    # Caspase-1 & IL-1beta
    casp1 = speck * 0.94
    il1b = tl.minimum(600.0, 15.0 + casp1 * 5.2)
    tl.store(il1b_out_ptr + offsets, il1b, mask=mask)

    # M1/M2 Phenotype ratio
    raw_ratio = (mac1 / 15.0) * (1.0 - (d * (0.88 / 100.0)))
    ratio = tl.maximum(0.12, tl.minimum(6.0, raw_ratio))
    tl.store(ratio_out_ptr + offsets, ratio, mask=mask)

def launch_microglial_mac1_inflammasome_triton(gamma, tlr4, p2x7, shed_dose):
    assert gamma.is_cuda and tlr4.is_cuda and p2x7.is_cuda and shed_dose.is_cuda
    n_elements = gamma.numel()
    mac1_out = torch.empty_like(gamma)
    speck_out = torch.empty_like(gamma)
    il1b_out = torch.empty_like(gamma)
    ratio_out = torch.empty_like(gamma)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_microglial_mac1_inflammasome_kernel[grid](
        gamma, tlr4, p2x7, shed_dose,
        mac1_out, speck_out, il1b_out, ratio_out,
        n_elements,
        BLOCK_SIZE=1024
    )
    return mac1_out, speck_out, il1b_out, ratio_out
