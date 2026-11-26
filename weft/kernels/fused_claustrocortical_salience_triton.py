"""
WEFT Fused Claustrocortical Salience Synchronization Kernel (Triton 3.9).
Milestone Paper #150 Series.
Evaluates cross-modal salience routing and widespread feedforward GABAergic suppression
of non-salient cortical columns:
I_eff[i] = (1.0 + gamma_cl * S_cue[i]) * I_stim[i] - (alpha_gaba * G_cl) * (1.0 - S_cue[i]) * I_stim[i]
directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_claustrocortical_salience_kernel(
    i_stim_ptr,     # [B, N] input sensory drive across modalities
    s_cue_ptr,      # [B, N] salience mask [0 or 1]
    i_eff_ptr,      # [B, N] output effective synchronized drive
    g_cl_ptr,       # [B] global claustrum gain scalar in [0, 1]
    gamma_cl: tl.constexpr,   # amplification for salient channels
    alpha_gaba: tl.constexpr, # feedforward suppression for non-salient channels
    N_TOTAL: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N_TOTAL:
        return

    stim = tl.load(i_stim_ptr + pid).to(tl.float32)
    cue = tl.load(s_cue_ptr + pid).to(tl.float32)
    g_cl = tl.load(g_cl_ptr + 0).to(tl.float32)

    # If cue is salient (cue > 0.5): amplify drive and phase-align
    # If non-salient: suppress via feedforward inhibition
    gain = 0.0
    if cue > 0.5:
        gain = 1.0 + gamma_cl * g_cl
    else:
        gain = 1.0 - alpha_gaba * g_cl
        if gain < 0.05:
            gain = 0.05

    out = stim * gain
    tl.store(i_eff_ptr + pid, out.to(tl.float16))

def apply_claustrocortical_salience_triton(
    i_stim: torch.Tensor,
    s_cue: torch.Tensor,
    i_eff: torch.Tensor,
    g_cl: torch.Tensor,
    gamma_cl: float = 0.8,
    alpha_gaba: float = 0.85
) -> torch.Tensor:
    """Evaluates claustrocortical salience routing on GPU."""
    assert i_stim.is_cuda and s_cue.is_cuda and i_eff.is_cuda
    N_TOTAL = i_stim.numel()
    grid = (N_TOTAL,)
    _fused_claustrocortical_salience_kernel[grid](
        i_stim, s_cue, i_eff, g_cl,
        gamma_cl=float(gamma_cl),
        alpha_gaba=float(alpha_gaba),
        N_TOTAL=N_TOTAL
    )
    return i_eff

def compute_claustrocortical_salience_step(
    stim: float,
    cue: float,
    g_cl: float,
    gamma_cl: float = 0.8,
    alpha_gaba: float = 0.85
) -> float:
    """Analytical CPU evaluation of claustrum salience gating."""
    if cue > 0.5:
        gain = 1.0 + gamma_cl * g_cl
    else:
        gain = max(0.05, 1.0 - alpha_gaba * g_cl)
    return stim * gain
