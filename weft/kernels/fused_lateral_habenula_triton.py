"""
WEFT Fused Lateral Habenula Aversion & Dopaminergic Dip Kernel (Triton 3.9).
Milestone Paper #154 Series.
Models:
1. Negative Reward Prediction Error and Aversive Input to Lateral Habenula (LHb):
   I_lhb = max(0.0, -RPE * beta_rpe) + I_aversive
2. Disynaptic GABAergic inhibition via RMTg onto midbrain SNc/VTA Dopamine:
   DA_out = clamp(DA_base + alpha_da * RPE - gamma_rmtg * I_lhb, 0.0, 1.0)
directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_lateral_habenula_kernel(
    rpe_ptr,        # [B] reward prediction errors (can be negative)
    aversive_ptr,   # [B, N] direct aversive / pain inputs
    lhb_out_ptr,    # [B, N] output LHb burst activation
    da_out_ptr,     # [B] output midbrain dopamine firing rate
    da_base: tl.constexpr,     # baseline tonic dopamine e.g. 0.50
    beta_rpe: tl.constexpr,    # sensitivity to negative RPE e.g. 1.50
    gamma_rmtg: tl.constexpr,  # RMTg inhibitory clamp e.g. 0.40
    N_TOTAL: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N_TOTAL:
        return

    rpe = tl.load(rpe_ptr + 0).to(tl.float32)
    aversive = tl.load(aversive_ptr + pid).to(tl.float32)

    # Negative RPE triggers LHb bursting
    rpe_loss = 0.0
    if rpe < 0.0:
        rpe_loss = -rpe * beta_rpe

    lhb_act = rpe_loss + aversive * 0.8
    if lhb_act > 10.0:
        lhb_act = 10.0

    # Dopamine suppression via RMTg
    # DA = clamp(DA_base + (rpe > 0 ? rpe*0.5 : 0) - gamma_rmtg * lhb_act, 0.0, 1.0)
    da = da_base
    if rpe > 0.0:
        da = da + rpe * 0.5
    da = da - gamma_rmtg * lhb_act
    if da < 0.0:
        da = 0.0
    if da > 1.0:
        da = 1.0

    tl.store(lhb_out_ptr + pid, lhb_act.to(tl.float16))
    if pid == 0:
        tl.store(da_out_ptr + 0, da.to(tl.float16))

def apply_lateral_habenula_triton(
    rpe: torch.Tensor,
    aversive: torch.Tensor,
    lhb_out: torch.Tensor,
    da_out: torch.Tensor,
    da_base: float = 0.50,
    beta_rpe: float = 1.50,
    gamma_rmtg: float = 0.40
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates LHb anti-reward dynamics on GPU."""
    assert rpe.is_cuda and aversive.is_cuda
    N_TOTAL = aversive.numel()
    grid = (N_TOTAL,)
    _fused_lateral_habenula_kernel[grid](
        rpe, aversive, lhb_out, da_out,
        da_base=float(da_base),
        beta_rpe=float(beta_rpe),
        gamma_rmtg=float(gamma_rmtg),
        N_TOTAL=N_TOTAL
    )
    return lhb_out, da_out

def compute_habenular_step(
    rpe: float,
    aversive: float,
    da_base: float = 0.50,
    beta_rpe: float = 1.50,
    gamma_rmtg: float = 0.40
) -> tuple[float, float]:
    """Analytical CPU evaluation of LHb activation and DA dip."""
    rpe_loss = -rpe * beta_rpe if rpe < 0.0 else 0.0
    lhb_act = min(10.0, rpe_loss + aversive * 0.8)
    da = da_base + (rpe * 0.5 if rpe > 0.0 else 0.0) - gamma_rmtg * lhb_act
    da_clamped = max(0.0, min(1.0, da))
    return lhb_act, da_clamped
