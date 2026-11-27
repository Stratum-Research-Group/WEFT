"""
Fused Triton 3.9 Kernel: RPE Photoreceptor Outer Segment (POS) Phagocytosis Kinetics
Triton GPU kernel computing Gas6/Protein S opsonization, avb5 integrin binding,
MerTK receptor autophosphorylation, diurnal POS clearance, and A2E lipofuscin burden.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rpe_phagocytosis_kernel(
    mertk_ptr,
    gas6_ptr,
    binding_out_ptr,
    clearance_out_ptr,
    a2e_out_ptr,
    survival_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m = tl.load(mertk_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(gas6_ptr + offsets, mask=mask, other=0.0)

    # 1. avb5 integrin binding: min(98.0, g * 1.05)
    bind = tl.minimum(98.0, g * 1.05)

    # 2. Phagocytosis rate: (m * bind) / 100.0
    rate = (m * bind) * 0.01

    # 3. Diurnal POS clearance: min(99.0, rate * 1.40)
    clear = tl.minimum(99.0, rate * 1.40)

    # 4. A2E lipofuscin accumulation from uncleared oxidized discs
    unclar = tl.maximum(0.0, 100.0 - clear)
    raw_a2e = unclar * 2.2
    a2e = tl.maximum(5.0, tl.minimum(95.0, raw_a2e))

    # 5. Photoreceptor survival probability
    loss = a2e * 0.75
    surv = tl.maximum(10.0, 100.0 - loss)

    tl.store(binding_out_ptr + offsets, bind, mask=mask)
    tl.store(clearance_out_ptr + offsets, clear, mask=mask)
    tl.store(a2e_out_ptr + offsets, a2e, mask=mask)
    tl.store(survival_out_ptr + offsets, surv, mask=mask)


def fused_rpe_phagocytosis_triton(
    mertk: torch.Tensor,
    gas6: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused RPE phagocytosis kernel on CUDA or returns CPU simulation.
    """
    n_elements = mertk.numel()
    binding = torch.empty_like(mertk)
    clearance = torch.empty_like(mertk)
    a2e = torch.empty_like(mertk)
    survival = torch.empty_like(mertk)

    if mertk.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_rpe_phagocytosis_kernel[grid](
            mertk,
            gas6,
            binding,
            clearance,
            a2e,
            survival,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        bind = torch.clamp(gas6 * 1.05, max=98.0)
        rate = (mertk * bind) * 0.01
        clear = torch.clamp(rate * 1.40, max=99.0)
        unclar = torch.clamp(100.0 - clear, min=0.0)
        raw_a2e = unclar * 2.2
        a2e = torch.clamp(raw_a2e, min=5.0, max=95.0)
        surv = torch.clamp(100.0 - a2e * 0.75, min=10.0)
        binding = bind
        clearance = clear
        survival = surv

    return binding, clearance, a2e, survival
