"""Triton 3.9 GPU kernel for fused VTA Dopaminergic Three-Factor Synaptic Plasticity.

Computes batched Hebbian eligibility trace accumulation, extracellular dopamine RPE
consolidation, and bounded synaptic efficacy updates across neural circuits.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_vta_plasticity_kernel(
    pre_rate_ptr,
    post_rate_ptr,
    dopamine_rpe_ptr,
    current_weight_ptr,
    out_eligibility_ptr,
    out_delta_w_ptr,
    out_updated_w_ptr,
    eta,
    min_w,
    max_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    pre = tl.load(pre_rate_ptr + offsets, mask=mask, other=0.0)
    post = tl.load(post_rate_ptr + offsets, mask=mask, other=0.0)
    rpe = tl.load(dopamine_rpe_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(current_weight_ptr + offsets, mask=mask, other=1.0)

    # Eligibility trace e = (pre * post) * 0.01
    trace = pre * post * 0.01

    # Synaptic delta dW = eta * trace * RPE
    dw = eta * trace * rpe

    # Updated weight W_next = clamp(W + dW, min_w, max_w)
    w_raw = w + dw
    w_clamped = tl.maximum(min_w, tl.minimum(max_w, w_raw))

    # Store results
    tl.store(out_eligibility_ptr + offsets, trace, mask=mask)
    tl.store(out_delta_w_ptr + offsets, dw, mask=mask)
    tl.store(out_updated_w_ptr + offsets, w_clamped, mask=mask)


def launch_fused_vta_plasticity(
    pre_rate: torch.Tensor,
    post_rate: torch.Tensor,
    dopamine_rpe: torch.Tensor,
    current_weight: torch.Tensor,
    eta: float = 0.08,
    min_w: float = 0.05,
    max_w: float = 3.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused VTA dopaminergic three-factor plasticity kernel."""
    assert pre_rate.is_contiguous()
    assert post_rate.is_contiguous()
    assert dopamine_rpe.is_contiguous()
    assert current_weight.is_contiguous()

    n_elements = pre_rate.numel()
    out_eligibility = torch.empty_like(pre_rate)
    out_delta_w = torch.empty_like(pre_rate)
    out_updated_w = torch.empty_like(pre_rate)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_vta_plasticity_kernel[grid](
        pre_rate,
        post_rate,
        dopamine_rpe,
        current_weight,
        out_eligibility,
        out_delta_w,
        out_updated_w,
        eta,
        min_w,
        max_w,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_eligibility, out_delta_w, out_updated_w
