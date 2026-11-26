"""Triton 3.9 GPU kernel for fused Basal Forebrain Acetylcholine (BF-ACh) attentional gating.

Computes batched cholinergic modulation of thalamocortical sensory feedforward gain
versus presynaptic suppression of intracortical recurrent feedback across cortical columns.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_basal_forebrain_ach_kernel(
    ach_tone_ptr,
    phasic_surge_ptr,
    feedforward_in_ptr,
    recurrent_in_ptr,
    out_g_ff_ptr,
    out_w_rec_ptr,
    out_response_ptr,
    out_cue_idx_ptr,
    alpha_ff,
    beta_rec,
    noise_base,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    tone = tl.load(ach_tone_ptr + offsets, mask=mask, other=0.0)
    surge = tl.load(phasic_surge_ptr + offsets, mask=mask, other=0.0)
    in_ff = tl.load(feedforward_in_ptr + offsets, mask=mask, other=0.0)
    in_rec = tl.load(recurrent_in_ptr + offsets, mask=mask, other=0.0)

    # Total effective cholinergic concentration
    eff_ach = tone + surge

    # Feedforward gain G_ff = 1.0 + alpha_ff * eff_ach
    g_ff = 1.0 + alpha_ff * eff_ach

    # Recurrent synaptic weight W_rec = 1.0 / (1.0 + beta_rec * eff_ach)
    w_rec_denom = 1.0 + beta_rec * eff_ach
    w_rec = 1.0 / w_rec_denom

    # Cortical ensemble response: Resp = G_ff * in_ff + W_rec * in_rec
    sig_ff = g_ff * in_ff
    sig_rec = w_rec * in_rec
    response = sig_ff + sig_rec

    # Cue detection index: sig_ff / (noise_base + sig_rec)
    denom = noise_base + sig_rec
    cue_idx = sig_ff / denom

    # Store results
    tl.store(out_g_ff_ptr + offsets, g_ff, mask=mask)
    tl.store(out_w_rec_ptr + offsets, w_rec, mask=mask)
    tl.store(out_response_ptr + offsets, response, mask=mask)
    tl.store(out_cue_idx_ptr + offsets, cue_idx, mask=mask)


def launch_fused_basal_forebrain_ach(
    ach_tone: torch.Tensor,
    phasic_surge: torch.Tensor,
    feedforward_in: torch.Tensor,
    recurrent_in: torch.Tensor,
    alpha_ff: float = 0.35,
    beta_rec: float = 0.50,
    noise_base: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused basal forebrain ACh attentional gating kernel."""
    assert ach_tone.is_contiguous()
    assert phasic_surge.is_contiguous()
    assert feedforward_in.is_contiguous()
    assert recurrent_in.is_contiguous()

    n_elements = ach_tone.numel()
    out_g_ff = torch.empty_like(ach_tone)
    out_w_rec = torch.empty_like(ach_tone)
    out_response = torch.empty_like(ach_tone)
    out_cue_idx = torch.empty_like(ach_tone)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_basal_forebrain_ach_kernel[grid](
        ach_tone,
        phasic_surge,
        feedforward_in,
        recurrent_in,
        out_g_ff,
        out_w_rec,
        out_response,
        out_cue_idx,
        alpha_ff,
        beta_rec,
        noise_base,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_g_ff, out_w_rec, out_response, out_cue_idx
