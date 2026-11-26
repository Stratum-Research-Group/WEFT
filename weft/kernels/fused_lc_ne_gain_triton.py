"""Triton 3.9 GPU kernel for fused Locus Coeruleus Norepinephrine (LC-NE) gain modulation.

Computes batched cortical transfer function gain scaling, non-linear tonic noise injection,
and signal-to-noise ratio (SNR) optimization across thalamocortical ensembles.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lc_ne_gain_kernel(
    tonic_ptr,
    phasic_ptr,
    salience_ptr,
    baseline_noise_ptr,
    out_gain_ptr,
    out_signal_ptr,
    out_noise_ptr,
    out_snr_ptr,
    alpha_gain,
    beta_noise,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    tonic = tl.load(tonic_ptr + offsets, mask=mask, other=0.0)
    phasic = tl.load(phasic_ptr + offsets, mask=mask, other=0.0)
    salience = tl.load(salience_ptr + offsets, mask=mask, other=0.0)
    base_noise = tl.load(baseline_noise_ptr + offsets, mask=mask, other=1.0)

    # Compute effective noradrenergic concentration
    effective_ne = tonic + phasic

    # Cortical gain G = 1.0 + alpha * effective_ne
    gain = 1.0 + alpha_gain * effective_ne

    # Amplified task signal S = G * salience
    signal = gain * salience

    # Cortical noise N = base_noise + beta * tonic
    noise = base_noise + beta_noise * tonic

    # SNR = max(0.01, signal / noise)
    snr = signal / noise
    snr = tl.maximum(snr, 0.01)

    # Store results
    tl.store(out_gain_ptr + offsets, gain, mask=mask)
    tl.store(out_signal_ptr + offsets, signal, mask=mask)
    tl.store(out_noise_ptr + offsets, noise, mask=mask)
    tl.store(out_snr_ptr + offsets, snr, mask=mask)


def launch_fused_lc_ne_gain(
    tonic: torch.Tensor,
    phasic: torch.Tensor,
    salience: torch.Tensor,
    baseline_noise: torch.Tensor,
    alpha_gain: float = 0.35,
    beta_noise: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused LC-NE cortical gain modulation kernel."""
    assert tonic.is_contiguous()
    assert phasic.is_contiguous()
    assert salience.is_contiguous()
    assert baseline_noise.is_contiguous()

    n_elements = tonic.numel()
    out_gain = torch.empty_like(tonic)
    out_signal = torch.empty_like(tonic)
    out_noise = torch.empty_like(tonic)
    out_snr = torch.empty_like(tonic)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_lc_ne_gain_kernel[grid](
        tonic,
        phasic,
        salience,
        baseline_noise,
        out_gain,
        out_signal,
        out_noise,
        out_snr,
        alpha_gain,
        beta_noise,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_gain, out_signal, out_noise, out_snr
