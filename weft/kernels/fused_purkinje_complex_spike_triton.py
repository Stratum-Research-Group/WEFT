"""
Fused Triton 3.9 Kernel: Cerebellar Purkinje Complex Spike & Climbing Fiber LTD Kinetics
Triton GPU kernel computing climbing fiber-driven dendritic Ca2+ transients, mGluR1/IP3/DAG co-activation,
parallel fiber AMPA receptor internalization (LTD), and motor prediction error calibration.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_purkinje_complex_spike_kernel(
    cf_freq_ptr,
    pf_drive_ptr,
    ca_out_ptr,
    mglur_out_ptr,
    ltd_out_ptr,
    error_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    cf = tl.load(cf_freq_ptr + offsets, mask=mask, other=0.0)
    pf = tl.load(pf_drive_ptr + offsets, mask=mask, other=0.0)

    # 1. Dendritic calcium transient (uM): cf * 0.55 + pf * 0.008
    ca = cf * 0.55 + pf * 0.008

    # 2. mGluR1/IP3/DAG activation ratio
    mglur = pf * 0.042

    # 3. Parallel fiber LTD (%): clamp((ca * mglur * 7.6), 5.0, 65.0)
    raw_ltd = ca * mglur * 7.6
    ltd = tl.maximum(5.0, tl.minimum(65.0, raw_ltd))

    # 4. Motor prediction error score
    err_raw = 1.0 - ltd * 0.017
    error = tl.maximum(0.05, err_raw)

    tl.store(ca_out_ptr + offsets, ca, mask=mask)
    tl.store(mglur_out_ptr + offsets, mglur, mask=mask)
    tl.store(ltd_out_ptr + offsets, ltd, mask=mask)
    tl.store(error_out_ptr + offsets, error, mask=mask)


def fused_purkinje_complex_spike_triton(
    cf_freq: torch.Tensor,
    pf_drive: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused Purkinje complex spike kernel on CUDA or returns CPU simulation.
    """
    n_elements = cf_freq.numel()
    ca = torch.empty_like(cf_freq)
    mglur = torch.empty_like(cf_freq)
    ltd = torch.empty_like(cf_freq)
    error = torch.empty_like(cf_freq)

    if cf_freq.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_purkinje_complex_spike_kernel[grid](
            cf_freq,
            pf_drive,
            ca,
            mglur,
            ltd,
            error,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        ca = cf_freq * 0.55 + pf_drive * 0.008
        mglur = pf_drive * 0.042
        raw_ltd = ca * mglur * 7.6
        ltd = torch.clamp(raw_ltd, min=5.0, max=65.0)
        error = torch.clamp(1.0 - ltd * 0.017, min=0.05)

    return ca, mglur, ltd, error
