"""
Fused Triton 3.9 Kernel: Target #399
Adaptive Closed-Loop DBS (aDBS): Phase-Specific Desynchronization & Beta-Burst Truncation
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _adbs_closed_loop_kernel(
    raw_duration_ptr,
    lfp_phase_ptr,
    truncated_duration_ptr,
    stim_intensity_ptr,
    beta_power_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    raw_dur = tl.load(raw_duration_ptr + offsets, mask=mask, other=100.0)
    phase = tl.load(lfp_phase_ptr + offsets, mask=mask, other=0.0)

    # Trigger threshold (250 ms)
    is_triggered = raw_dur > 250.0

    # Phase window: [150, 210] deg desynchronizing quadrant
    phase_norm = phase % 360.0
    is_opt_phase = (phase_norm >= 150.0) & (phase_norm <= 210.0)

    # Truncated duration
    trunc_dur = tl.where(is_opt_phase, 275.0, 325.0)
    final_dur = tl.where(is_triggered, trunc_dur, raw_dur)
    tl.store(truncated_duration_ptr + offsets, final_dur, mask=mask)

    # Stimulation intensity
    stim_val = tl.where(is_opt_phase, 2.5, 0.8)
    final_stim = tl.where(is_triggered, stim_val, 0.0)
    tl.store(stim_intensity_ptr + offsets, final_stim, mask=mask)

    # Beta power
    power = 20.0 + 0.12 * final_dur
    tl.store(beta_power_ptr + offsets, power, mask=mask)


def compute_adbs_closed_loop_triton(raw_duration: torch.Tensor, lfp_phase: torch.Tensor):
    """
    Python wrapper executing fused Triton aDBS kernel.
    """
    assert raw_duration.is_cuda == lfp_phase.is_cuda
    n_elements = raw_duration.numel()
    device = raw_duration.device

    truncated_dur = torch.empty(n_elements, device=device, dtype=torch.float32)
    stim_intensity = torch.empty(n_elements, device=device, dtype=torch.float32)
    beta_power = torch.empty(n_elements, device=device, dtype=torch.float32)

    if not raw_duration.is_cuda:
        # Fallback CPU execution
        is_triggered = raw_duration > 250.0
        phase_norm = lfp_phase % 360.0
        is_opt_phase = (phase_norm >= 150.0) & (phase_norm <= 210.0)
        trunc_dur = torch.where(is_opt_phase, 275.0, 325.0)
        final_dur = torch.where(is_triggered, trunc_dur, raw_duration)
        stim_val = torch.where(is_opt_phase, 2.5, 0.8)
        final_stim = torch.where(is_triggered, stim_val, 0.0)
        power = 20.0 + 0.12 * final_dur
        return final_dur, final_stim, power

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _adbs_closed_loop_kernel[grid](
        raw_duration,
        lfp_phase,
        truncated_dur,
        stim_intensity,
        beta_power,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return truncated_dur, stim_intensity, beta_power
