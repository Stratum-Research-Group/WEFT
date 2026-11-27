"""
Fused Triton 3.9 Kernel: Choroid Plexus CSF Secretion & Intracranial Pressure (ICP) Kinetics
Triton GPU kernel computing CA-II catalysis, NKCC1 cotransport, transepithelial osmotic gradient,
bulk volumetric CSF production, and intracranial compliance reserve under the Monro-Kellie doctrine.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_choroid_plexus_kernel(
    ca2_ptr,
    nkcc1_ptr,
    grad_out_ptr,
    csf_out_ptr,
    icp_out_ptr,
    comp_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ca = tl.load(ca2_ptr + offsets, mask=mask, other=0.0)
    nk = tl.load(nkcc1_ptr + offsets, mask=mask, other=0.0)

    # 1. Transepithelial osmotic gradient: 6.0 + ca * 0.105 + nk * 0.080
    grad = 6.0 + ca * 0.105 + nk * 0.080

    # 2. CSF production rate (mL/min): (grad / 20.0) * 0.36
    csf = (grad / 20.0) * 0.36

    # 3. Intracranial pressure (mmHg): 7.0 + (csf / 0.35) * 4.5
    icp = 7.0 + (csf / 0.35) * 4.5

    # 4. Compliance reserve (%): max(10.0, 100.0 - max(0.0, icp - 7.0) * 8.0)
    icp_excess = tl.maximum(0.0, icp - 7.0)
    comp = tl.maximum(10.0, 100.0 - icp_excess * 8.0)

    tl.store(grad_out_ptr + offsets, grad, mask=mask)
    tl.store(csf_out_ptr + offsets, csf, mask=mask)
    tl.store(icp_out_ptr + offsets, icp, mask=mask)
    tl.store(comp_out_ptr + offsets, comp, mask=mask)


def fused_choroid_plexus_triton(
    ca2: torch.Tensor,
    nkcc1: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused choroid plexus CSF secretion kernel on CUDA or returns CPU simulation.
    """
    n_elements = ca2.numel()
    grad = torch.empty_like(ca2)
    csf = torch.empty_like(ca2)
    icp = torch.empty_like(ca2)
    comp = torch.empty_like(ca2)

    if ca2.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_choroid_plexus_kernel[grid](
            ca2,
            nkcc1,
            grad,
            csf,
            icp,
            comp,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        grad = 6.0 + ca2 * 0.105 + nkcc1 * 0.080
        csf = (grad / 20.0) * 0.36
        icp = 7.0 + (csf / 0.35) * 4.5
        icp_excess = torch.clamp(icp - 7.0, min=0.0)
        comp = torch.clamp(100.0 - icp_excess * 8.0, min=10.0)

    return grad, csf, icp, comp
