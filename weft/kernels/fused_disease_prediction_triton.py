"""Triton 3.9 GPU kernel for fused Electrophysiological Disease State Prediction.

Computes batched LFP beta power emergence, phase-amplitude coupling (PAC),
aperiodic 1/f spectral tilt, and early predictive warning triggers on GPU tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_disease_prediction_kernel(
    da_ptr,
    dr_ptr,
    jit_ptr,
    out_beta_ptr,
    out_pac_ptr,
    out_slope_ptr,
    out_risk_ptr,
    out_alert_ptr,
    beta_base,
    beta_scale,
    beta_max,
    pac_th,
    pac_scale,
    slope_base,
    slope_da_scale,
    slope_jit_scale,
    slope_min,
    slope_max,
    w_beta,
    w_pac,
    w_slope,
    slope_ref,
    alert_th,
    alert_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    da = tl.load(da_ptr + offsets, mask=mask, other=1.0)
    dr = tl.load(dr_ptr + offsets, mask=mask, other=0.5)
    jit = tl.load(jit_ptr + offsets, mask=mask, other=0.0)

    # 1. Beta Power
    deficit = tl.maximum(0.0, 1.0 - da)
    def_sq = deficit * deficit
    dr_amp = 0.5 + 0.5 * dr
    raw_beta = beta_base + beta_scale * def_sq * dr_amp
    beta = tl.maximum(0.0, tl.minimum(beta_max, raw_beta))

    # 2. Phase-Amplitude Coupling (PAC)
    raw_pac = (beta - pac_th) * pac_scale
    pac = tl.maximum(0.0, tl.minimum(1.0, raw_pac))

    # 3. 1/f Spectral Tilt
    raw_slope = slope_base - slope_da_scale * deficit + slope_jit_scale * jit
    slope = tl.maximum(slope_min, tl.minimum(slope_max, raw_slope))

    # 4. Integrated Disease Risk Score
    inv_slope = tl.maximum(0.0, slope_ref - slope)
    raw_risk = w_beta * beta + w_pac * pac + w_slope * inv_slope
    risk = tl.maximum(0.0, tl.minimum(1.0, raw_risk))

    # 5. Predictive Alert
    raw_alert = 0.5 + alert_slope * (risk - alert_th)
    alert = tl.maximum(0.0, tl.minimum(1.0, raw_alert))

    # Store results
    tl.store(out_beta_ptr + offsets, beta, mask=mask)
    tl.store(out_pac_ptr + offsets, pac, mask=mask)
    tl.store(out_slope_ptr + offsets, slope, mask=mask)
    tl.store(out_risk_ptr + offsets, risk, mask=mask)
    tl.store(out_alert_ptr + offsets, alert, mask=mask)


def launch_fused_disease_prediction(
    dopamine: torch.Tensor,
    drive: torch.Tensor,
    jitter: torch.Tensor,
    beta_base: float = 10.0,
    beta_scale: float = 80.0,
    beta_max: float = 100.0,
    pac_th: float = 15.0,
    pac_scale: float = 1.0 / 75.0,
    slope_base: float = 2.8,
    slope_da_scale: float = 1.4,
    slope_jit_scale: float = 0.2,
    slope_min: float = 1.0,
    slope_max: float = 3.5,
    w_beta: float = 0.50 / 90.0,
    w_pac: float = 0.35,
    w_slope: float = 0.15 / 2.0,
    slope_ref: float = 3.0,
    alert_th: float = 0.45,
    alert_slope: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused electrophysiological disease prediction kernel."""
    assert dopamine.is_contiguous()
    assert drive.is_contiguous()
    assert jitter.is_contiguous()

    n_elements = dopamine.numel()
    out_beta = torch.empty_like(dopamine)
    out_pac = torch.empty_like(dopamine)
    out_slope = torch.empty_like(dopamine)
    out_risk = torch.empty_like(dopamine)
    out_alert = torch.empty_like(dopamine)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_disease_prediction_kernel[grid](
        dopamine,
        drive,
        jitter,
        out_beta,
        out_pac,
        out_slope,
        out_risk,
        out_alert,
        beta_base,
        beta_scale,
        beta_max,
        pac_th,
        pac_scale,
        slope_base,
        slope_da_scale,
        slope_jit_scale,
        slope_min,
        slope_max,
        w_beta,
        w_pac,
        w_slope,
        slope_ref,
        alert_th,
        alert_slope,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_beta, out_pac, out_slope, out_risk, out_alert
