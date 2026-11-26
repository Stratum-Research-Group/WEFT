"""
Triton 3.9 Fused Three-Factor Corticostriatal Plasticity Kernel.
Calculates:
1. Direct D1 dSPN "Go" pathway facilitation under positive dopamine bursts.
2. Indirect D2 iSPN "NoGo" pathway suppression under dopamine transients.
3. Opponent contrast and logistic action selection execution probability.
Target #381 in WEFT Verification Suite.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_striatal_dopamine_kernel(
    cortex_ptr,
    da_rpe_ptr,
    d1_go_out_ptr,
    d2_nogo_out_ptr,
    action_prob_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c = tl.load(cortex_ptr + offsets, mask=mask, other=0.0)
    da = tl.load(da_rpe_ptr + offsets, mask=mask, other=0.0)

    # D1 Go drive: c * max(0.1, 1.0 + da * 0.8)
    d1_mod = tl.maximum(0.1, 1.0 + da * 0.8)
    d1 = c * d1_mod
    tl.store(d1_go_out_ptr + offsets, d1, mask=mask)

    # D2 NoGo drive: c * max(0.1, 1.0 - da * 0.7)
    d2_mod = tl.maximum(0.1, 1.0 - da * 0.7)
    d2 = c * d2_mod
    tl.store(d2_nogo_out_ptr + offsets, d2, mask=mask)

    # Net opponent contrast: (d1 - d2) / 25.0
    diff = (d1 - d2) / 25.0
    # Rational sigmoid approximation: 0.5 + 0.5 * (diff / (1.0 + abs(diff)))
    prob = 0.5 + 0.5 * (diff / (1.0 + tl.abs(diff)))
    tl.store(action_prob_out_ptr + offsets, prob, mask=mask)

def launch_striatal_dopamine_transform(
    cortex: torch.Tensor,
    da_rpe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = cortex.numel()
    d1_out = torch.empty_like(cortex)
    d2_out = torch.empty_like(cortex)
    prob_out = torch.empty_like(cortex)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_striatal_dopamine_kernel[grid](
        cortex,
        da_rpe,
        d1_out,
        d2_out,
        prob_out,
        n,
        BLOCK_SIZE=256,
    )
    return d1_out, d2_out, prob_out
