"""
Fused Triton 3.9 Kernel: Vagal Cholinergic Anti-Inflammatory Reflex Kinetics
Triton GPU kernel computing splenic ACh release, macrophage alpha7nAChR binding,
STAT3 phosphorylation, and nuclear NF-kB translocation suppression.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_vagal_anti_inflammatory_kernel(
    vagus_freq_ptr,
    lps_insult_ptr,
    ach_out_ptr,
    alpha7_occ_ptr,
    stat3_out_ptr,
    nfkb_out_ptr,
    tnf_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    freq = tl.load(vagus_freq_ptr + offsets, mask=mask, other=0.0)
    lps = tl.load(lps_insult_ptr + offsets, mask=mask, other=0.0)

    # 1. Splenic acetylcholine release: min(600.0, freq * 24.5)
    ach_raw = freq * 24.5
    ach = tl.minimum(600.0, ach_raw)

    # 2. alpha7nAChR receptor occupancy: (ach / (ach + 120.0)) * 100.0
    denom = ach + 120.0
    occ = (ach / denom) * 100.0

    # 3. STAT3 phosphorylation ratio: occ * 0.045
    stat3 = occ * 0.045

    # 4. Nuclear NF-kB translocation: max(8.0, min(100.0, lps - occ * 0.82))
    suppr = occ * 0.82
    nfkb_raw = lps - suppr
    nfkb = tl.maximum(8.0, tl.minimum(100.0, nfkb_raw))

    # 5. Plasma TNF-alpha: 15.0 + nfkb * 1.4
    tnf = 15.0 + nfkb * 1.4

    tl.store(ach_out_ptr + offsets, ach, mask=mask)
    tl.store(alpha7_occ_ptr + offsets, occ, mask=mask)
    tl.store(stat3_out_ptr + offsets, stat3, mask=mask)
    tl.store(nfkb_out_ptr + offsets, nfkb, mask=mask)
    tl.store(tnf_out_ptr + offsets, tnf, mask=mask)


def fused_vagal_anti_inflammatory_triton(
    vagus_freq: torch.Tensor,
    lps_insult: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused vagal anti-inflammatory kernel on CUDA or returns CPU simulation.
    """
    n_elements = vagus_freq.numel()
    ach = torch.empty_like(vagus_freq)
    occ = torch.empty_like(vagus_freq)
    stat3 = torch.empty_like(vagus_freq)
    nfkb = torch.empty_like(vagus_freq)
    tnf = torch.empty_like(vagus_freq)

    if vagus_freq.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_vagal_anti_inflammatory_kernel[grid](
            vagus_freq,
            lps_insult,
            ach,
            occ,
            stat3,
            nfkb,
            tnf,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        ach = torch.clamp(vagus_freq * 24.5, max=600.0)
        occ = (ach / (ach + 120.0)) * 100.0
        stat3 = occ * 0.045
        nfkb = torch.clamp(lps_insult - occ * 0.82, min=8.0, max=100.0)
        tnf = 15.0 + nfkb * 1.4

    return ach, occ, stat3, nfkb, tnf
