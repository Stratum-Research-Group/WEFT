"""
Fused Triton 3.9 Kernel: Striatal MSN D1/D2 Dopaminergic Plasticity Kinetics
Triton GPU kernel computing D1 Golf/PKA phosphorylation, D2 Gi/Calcineurin balance,
direct vs indirect pathway gains, and motor initiation gain ratio.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_striatal_d1_d2_kernel(
    da_ptr,
    glut_ptr,
    pka_out_ptr,
    cn_out_ptr,
    d1_out_ptr,
    d2_out_ptr,
    ratio_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    da = tl.load(da_ptr + offsets, mask=mask, other=0.0)
    glu = tl.load(glut_ptr + offsets, mask=mask, other=0.0)

    # 1. D1 PKA ratio: 0.5 + (da / (da + 600.0)) * 4.5
    denom_d1 = da + 600.0
    pka = 0.5 + (da / denom_d1) * 4.5

    # 2. D2 Calcineurin ratio: 4.5 - (da / (da + 150.0)) * 3.5
    denom_d2 = da + 150.0
    cn = 4.5 - (da / denom_d2) * 3.5

    # 3. Direct (D1) and Indirect (D2) pathway gains
    d1 = (glu / 100.0) * pka * 0.75
    d2 = (glu / 100.0) * cn * 0.45

    # 4. Gain ratio
    d2_safe = tl.maximum(0.01, d2)
    ratio = d1 / d2_safe

    tl.store(pka_out_ptr + offsets, pka, mask=mask)
    tl.store(cn_out_ptr + offsets, cn, mask=mask)
    tl.store(d1_out_ptr + offsets, d1, mask=mask)
    tl.store(d2_out_ptr + offsets, d2, mask=mask)
    tl.store(ratio_out_ptr + offsets, ratio, mask=mask)


def fused_striatal_d1_d2_triton(
    da: torch.Tensor,
    glut: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused striatal D1/D2 plasticity kernel on CUDA or returns CPU simulation.
    """
    n_elements = da.numel()
    pka = torch.empty_like(da)
    cn = torch.empty_like(da)
    d1 = torch.empty_like(da)
    d2 = torch.empty_like(da)
    ratio = torch.empty_like(da)

    if da.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_striatal_d1_d2_kernel[grid](
            da,
            glut,
            pka,
            cn,
            d1,
            d2,
            ratio,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        pka = 0.5 + (da / (da + 600.0)) * 4.5
        cn = 4.5 - (da / (da + 150.0)) * 3.5
        d1 = (glut / 100.0) * pka * 0.75
        d2 = (glut / 100.0) * cn * 0.45
        ratio = d1 / torch.clamp(d2, min=0.01)

    return pka, cn, d1, d2, ratio
