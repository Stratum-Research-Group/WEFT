"""
Fused Triton 3.9 Kernel: Presynaptic SNARE Complex Assembly & Munc18-1 Priming
Triton GPU kernel computing Munc13-mediated open syntaxin transitions, SNARE zippering,
readily releasable pool (RRP) occupancy, and sub-millisecond quantal release jitter.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_snare_priming_kernel(
    munc13_ptr,
    ca_ptr,
    open_syn_out_ptr,
    snare_zip_out_ptr,
    rrp_out_ptr,
    jitter_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m13 = tl.load(munc13_ptr + offsets, mask=mask, other=0.0)
    ca = tl.load(ca_ptr + offsets, mask=mask, other=1.0)

    # 1. Open Syntaxin: min(98.0, m13 * 0.98)
    open_syn = tl.minimum(98.0, m13 * 0.98)

    # 2. SNARE core zippering: (open_syn / (open_syn + 10.0)) * 100.0
    denom = open_syn + 10.0
    zip_val = (open_syn / denom) * 100.0

    # 3. RRP vesicle occupancy: zip * 0.95
    rrp = zip_val * 0.95

    # 4. Quantal release jitter: max(120.0, 1800.0 / ca + max(0.0, 80.0 - rrp) * 12.0)
    base_jit = 1800.0 / tl.maximum(1.0, ca)
    penalty = tl.maximum(0.0, 80.0 - rrp) * 12.0
    jitter = tl.maximum(120.0, base_jit + penalty)

    tl.store(open_syn_out_ptr + offsets, open_syn, mask=mask)
    tl.store(snare_zip_out_ptr + offsets, zip_val, mask=mask)
    tl.store(rrp_out_ptr + offsets, rrp, mask=mask)
    tl.store(jitter_out_ptr + offsets, jitter, mask=mask)


def fused_snare_priming_triton(
    munc13: torch.Tensor,
    ca: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused SNARE priming kernel on CUDA or returns CPU simulation.
    """
    n_elements = munc13.numel()
    open_syn = torch.empty_like(munc13)
    zip_val = torch.empty_like(munc13)
    rrp = torch.empty_like(munc13)
    jitter = torch.empty_like(munc13)

    if munc13.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_snare_priming_kernel[grid](
            munc13,
            ca,
            open_syn,
            zip_val,
            rrp,
            jitter,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        open_syn = torch.clamp(munc13 * 0.98, max=98.0)
        zip_val = (open_syn / (open_syn + 10.0)) * 100.0
        rrp = zip_val * 0.95
        base_jit = 1800.0 / torch.clamp(ca, min=1.0)
        penalty = torch.clamp(80.0 - rrp, min=0.0) * 12.0
        jitter = torch.clamp(base_jit + penalty, min=120.0)

    return open_syn, zip_val, rrp, jitter
