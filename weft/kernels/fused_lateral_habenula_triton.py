"""Triton 3.9 GPU kernel for fused Lateral Habenula (LHb) anti-reward and dopamine pause computation.

Computes batched negative reward prediction errors, LHb burst activation,
RMTg GABAergic transmission, and VTA dopaminergic pause gating.
"""

import torch
import triton
import triton.language as tl


def compute_habenular_step(rpe: float, aversive: float = 0.0, da_base: float = 0.5, beta_rpe: float = 1.5, gamma_rmtg: float = 0.4) -> tuple[float, float]:
    """Scalar reference step for Lateral Habenula - RMTg dopamine suppression testing."""
    if rpe < 0.0:
        nrpe = -rpe
        lhb_act = beta_rpe * nrpe + 0.8 * aversive
        da_suppression = gamma_rmtg * lhb_act
        da_out = max(0.0, da_base - da_suppression)
    else:
        lhb_act = 0.0
        da_out = min(1.0, da_base + 0.5 * rpe)
    return float(lhb_act), float(da_out)


@triton.jit
def _fused_lateral_habenula_kernel(
    exp_reward_ptr,
    act_outcome_ptr,
    stress_ptr,
    tonic_base_ptr,
    out_nrpe_ptr,
    out_lhb_ptr,
    out_rmtg_ptr,
    out_da_ptr,
    c_nrpe,
    c_stress,
    c_rmtg_base,
    c_rmtg_slope,
    c_da_slope,
    da_max,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    exp_rew = tl.load(exp_reward_ptr + offsets, mask=mask, other=0.0)
    act_out = tl.load(act_outcome_ptr + offsets, mask=mask, other=0.0)
    stress = tl.load(stress_ptr + offsets, mask=mask, other=0.0)
    base = tl.load(tonic_base_ptr + offsets, mask=mask, other=8.0)

    # Negative RPE = max(0.0, exp_rew - act_out)
    diff = exp_rew - act_out
    nrpe = tl.maximum(0.0, diff)

    # LHb Firing Rate = base + c_nrpe * nrpe + c_stress * stress
    lhb_rate = base + c_nrpe * nrpe + c_stress * stress

    # RMTg GABA = c_rmtg_base + c_rmtg_slope * lhb_rate
    rmtg_gaba = c_rmtg_base + c_rmtg_slope * lhb_rate

    # VTA DA = max(0.0, da_max - c_da_slope * (rmtg_gaba - 10.0))
    rmtg_delta = rmtg_gaba - 10.0
    da_raw = da_max - c_da_slope * rmtg_delta
    da_rate = tl.maximum(0.0, da_raw)

    # Store results
    tl.store(out_nrpe_ptr + offsets, nrpe, mask=mask)
    tl.store(out_lhb_ptr + offsets, lhb_rate, mask=mask)
    tl.store(out_rmtg_ptr + offsets, rmtg_gaba, mask=mask)
    tl.store(out_da_ptr + offsets, da_rate, mask=mask)


def launch_fused_lateral_habenula(
    expected_reward: torch.Tensor,
    actual_outcome: torch.Tensor,
    aversive_stress: torch.Tensor,
    tonic_baseline: torch.Tensor,
    c_nrpe: float = 0.45,
    c_stress: float = 0.35,
    c_rmtg_base: float = 5.0,
    c_rmtg_slope: float = 0.85,
    c_da_slope: float = 0.075,
    da_max: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused lateral habenula anti-reward and dopamine pause kernel."""
    assert expected_reward.is_contiguous()
    assert actual_outcome.is_contiguous()
    assert aversive_stress.is_contiguous()
    assert tonic_baseline.is_contiguous()

    n_elements = expected_reward.numel()
    out_nrpe = torch.empty_like(expected_reward)
    out_lhb = torch.empty_like(expected_reward)
    out_rmtg = torch.empty_like(expected_reward)
    out_da = torch.empty_like(expected_reward)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_lateral_habenula_kernel[grid](
        expected_reward,
        actual_outcome,
        aversive_stress,
        tonic_baseline,
        out_nrpe,
        out_lhb,
        out_rmtg,
        out_da,
        c_nrpe,
        c_stress,
        c_rmtg_base,
        c_rmtg_slope,
        c_da_slope,
        da_max,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_nrpe, out_lhb, out_rmtg, out_da
