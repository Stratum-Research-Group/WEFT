"""Triton 3.9 GPU kernel for fused Claustrum Cortical Synchronization & Salience Switching.

Computes batched claustral burst integration, widespread feedforward cortical inhibition
via fast-spiking interneurons, and cross-columnar phase reset synchrony.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_claustrum_synchronization_kernel(
    sal_in_ptr,
    task_drive_ptr,
    base_ptr,
    gain_ptr,
    out_rate_ptr,
    out_inh_ptr,
    out_sync_ptr,
    out_prob_ptr,
    c_sal,
    c_inh_scale,
    sync_base,
    sync_scale,
    c_drive_scale,
    c_inh_eff,
    inh_offset,
    sig_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    sal = tl.load(sal_in_ptr + offsets, mask=mask, other=0.0)
    task = tl.load(task_drive_ptr + offsets, mask=mask, other=0.0)
    base = tl.load(base_ptr + offsets, mask=mask, other=2.0)
    gain = tl.load(gain_ptr + offsets, mask=mask, other=0.5)

    # 1. Rate_cl = base + c_sal * sal
    rate = base + c_sal * sal

    # 2. Feedforward Inh = 1.0 + gain * c_inh_scale * rate
    gain_eff = gain * c_inh_scale
    inh = 1.0 + gain_eff * rate

    # 3. Synchrony = clamp(sync_base + sync_scale * rate, 0.0, 1.0)
    sync_raw = sync_base + sync_scale * rate
    sync = tl.maximum(0.0, tl.minimum(1.0, sync_raw))

    # 4. Switch Prob Proxy
    diff = sal - task
    net_drive = diff * c_drive_scale
    inh_contrib = (inh - inh_offset) * c_inh_eff
    comb = net_drive + inh_contrib
    prob_raw = 0.5 + sig_slope * comb
    prob = tl.maximum(0.0, tl.minimum(1.0, prob_raw))

    # Store results
    tl.store(out_rate_ptr + offsets, rate, mask=mask)
    tl.store(out_inh_ptr + offsets, inh, mask=mask)
    tl.store(out_sync_ptr + offsets, sync, mask=mask)
    tl.store(out_prob_ptr + offsets, prob, mask=mask)


def launch_fused_claustrum_synchronization(
    salience_input: torch.Tensor,
    task_drive: torch.Tensor,
    claustral_baseline: torch.Tensor,
    interneuron_gain: torch.Tensor,
    c_sal: float = 0.85,
    c_inh_scale: float = 1.80,
    sync_base: float = 0.10,
    sync_scale: float = 0.065,
    c_drive_scale: float = 0.80,
    c_inh_eff: float = 0.25,
    inh_offset: float = 3.0,
    sig_slope: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused claustrum synchronization and salience switching kernel."""
    assert salience_input.is_contiguous()
    assert task_drive.is_contiguous()
    assert claustral_baseline.is_contiguous()
    assert interneuron_gain.is_contiguous()

    n_elements = salience_input.numel()
    out_rate = torch.empty_like(salience_input)
    out_inh = torch.empty_like(salience_input)
    out_sync = torch.empty_like(salience_input)
    out_prob = torch.empty_like(salience_input)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_claustrum_synchronization_kernel[grid](
        salience_input,
        task_drive,
        claustral_baseline,
        interneuron_gain,
        out_rate,
        out_inh,
        out_sync,
        out_prob,
        c_sal,
        c_inh_scale,
        sync_base,
        sync_scale,
        c_drive_scale,
        c_inh_eff,
        inh_offset,
        sig_slope,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_rate, out_inh, out_sync, out_prob
