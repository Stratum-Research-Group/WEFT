"""
Fused Triton 3.9 Kernel: Microglial Complement-Mediated Synaptic Pruning Kinetics
Triton GPU kernel computing CD47 checkpoint suppression of C1q/C3b opsonization,
CR3 phagocytic engulfment, and dendritic spine survival probability.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_microglial_pruning_kernel(
    cd47_ptr,
    c1q_ptr,
    c3b_out_ptr,
    cr3_out_ptr,
    survival_out_ptr,
    ei_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    cd47 = tl.load(cd47_ptr + offsets, mask=mask, other=0.0)
    c1q = tl.load(c1q_ptr + offsets, mask=mask, other=0.0)

    # 1. CD47 suppression of C3b opsonization: (c1q * 1.2) * (1.0 - cd47 * 0.0085)
    suppr = cd47 * 0.0085
    c1q_prod = c1q * 1.2
    c3b_raw = c1q_prod * (1.0 - suppr)
    c3b = tl.maximum(5.0, tl.minimum(100.0, c3b_raw))

    # 2. CR3 engagement: (c3b / (c3b + 25.0)) * 100.0
    denom = c3b + 25.0
    cr3 = (c3b / denom) * 100.0

    # 3. Spine survival: max(10.0, min(100.0, 100.0 - cr3 * 0.85))
    loss = cr3 * 0.85
    survival = tl.maximum(10.0, tl.minimum(100.0, 100.0 - loss))

    # 4. Cortical E/I balance ratio: 1.15 * (survival / 85.0)
    ei = 1.15 * (survival / 85.0)

    tl.store(c3b_out_ptr + offsets, c3b, mask=mask)
    tl.store(cr3_out_ptr + offsets, cr3, mask=mask)
    tl.store(survival_out_ptr + offsets, survival, mask=mask)
    tl.store(ei_out_ptr + offsets, ei, mask=mask)


def fused_microglial_pruning_triton(
    cd47: torch.Tensor,
    c1q: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused microglial synaptic pruning kernel on CUDA or returns CPU simulation.
    """
    n_elements = cd47.numel()
    c3b = torch.empty_like(cd47)
    cr3 = torch.empty_like(cd47)
    survival = torch.empty_like(cd47)
    ei = torch.empty_like(cd47)

    if cd47.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_microglial_pruning_kernel[grid](
            cd47,
            c1q,
            c3b,
            cr3,
            survival,
            ei,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        suppr = cd47 * 0.0085
        c3b_raw = (c1q * 1.2) * (1.0 - suppr)
        c3b = torch.clamp(c3b_raw, min=5.0, max=100.0)
        cr3 = (c3b / (c3b + 25.0)) * 100.0
        survival = torch.clamp(100.0 - cr3 * 0.85, min=10.0, max=100.0)
        ei = 1.15 * (survival / 85.0)

    return c3b, cr3, survival, ei
