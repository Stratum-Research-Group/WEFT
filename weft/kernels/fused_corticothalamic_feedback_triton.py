"""
WEFT Fused Corticothalamic Layer VI Modulatory Feedback Kernel (Triton 3.9).
Milestone Paper #152 Series.
Models dual-target Layer VI feedback:
1. Center amplification: I_center_out = (1.0 + gamma_l6 * L6_drive) * I_center
2. Surround disynaptic inhibition via TRN: I_surround_out = max(0.05, 1.0 - alpha_trn * L6_drive) * I_surround
directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_corticothalamic_feedback_kernel(
    i_center_ptr,   # [B, N] receptive field center inputs
    i_surround_ptr, # [B, N] receptive field surround inputs
    out_center_ptr, # [B, N] output sharpened center drive
    out_surround_ptr,# [B, N] output suppressed surround drive
    l6_drive_ptr,   # [B] Layer VI cortical feedback drive in [0, 1]
    gamma_l6: tl.constexpr,   # Center modulatory gain boost e.g. 0.45
    alpha_trn: tl.constexpr,  # TRN surround inhibitory gain e.g. 0.80
    N_TOTAL: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N_TOTAL:
        return

    c_in = tl.load(i_center_ptr + pid).to(tl.float32)
    s_in = tl.load(i_surround_ptr + pid).to(tl.float32)
    l6 = tl.load(l6_drive_ptr + 0).to(tl.float32)

    # Center modulatory facilitation (mGluR1/NMDA kinetics)
    gain_c = 1.0 + gamma_l6 * l6
    c_out = c_in * gain_c

    # Surround disynaptic inhibition via Thalamic Reticular Nucleus (TRN)
    gain_s = 1.0 - alpha_trn * l6
    if gain_s < 0.05:
        gain_s = 0.05
    s_out = s_in * gain_s

    tl.store(out_center_ptr + pid, c_out.to(tl.float16))
    tl.store(out_surround_ptr + pid, s_out.to(tl.float16))

def apply_corticothalamic_feedback_triton(
    i_center: torch.Tensor,
    i_surround: torch.Tensor,
    out_center: torch.Tensor,
    out_surround: torch.Tensor,
    l6_drive: torch.Tensor,
    gamma_l6: float = 0.45,
    alpha_trn: float = 0.80
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates corticothalamic Layer VI receptive field tuning on GPU."""
    assert i_center.is_cuda and i_surround.is_cuda
    N_TOTAL = i_center.numel()
    grid = (N_TOTAL,)
    _fused_corticothalamic_feedback_kernel[grid](
        i_center, i_surround, out_center, out_surround, l6_drive,
        gamma_l6=float(gamma_l6),
        alpha_trn=float(alpha_trn),
        N_TOTAL=N_TOTAL
    )
    return out_center, out_surround

def compute_corticothalamic_feedback_step(
    c_in: float,
    s_in: float,
    l6: float,
    gamma_l6: float = 0.45,
    alpha_trn: float = 0.80
) -> tuple[float, float]:
    """Analytical CPU evaluation of Layer VI receptive field step."""
    c_out = c_in * (1.0 + gamma_l6 * l6)
    s_out = s_in * max(0.05, 1.0 - alpha_trn * l6)
    return c_out, s_out
