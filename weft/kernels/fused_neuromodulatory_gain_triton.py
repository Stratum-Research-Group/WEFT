"""
WEFT Fused Tri-Monoaminergic Gain Modulation Kernel (Triton 3.9).
Milestone Paper #149 Series.
Multiplicatively modulates cortical pyramidal neuronal gain and signal-to-noise ratio:
I_eff = (gamma_base + gamma_NE * NE) * I_feedforward + (1.0 - beta_ACh * ACh) * I_recurrent
with homeostatic saturation clamping directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_neuromodulatory_gain_kernel(
    i_feedforward_ptr, # [B, N] feedforward sensory current
    i_recurrent_ptr,   # [B, N] recurrent intracortical current
    i_eff_ptr,         # [B, N] output effective current
    ne_ptr,            # [B] Norepinephrine scalar in [0, 1]
    ach_ptr,           # [B] Acetylcholine scalar in [0, 1]
    gamma_base: tl.constexpr,
    gamma_ne: tl.constexpr,
    beta_ach: tl.constexpr,
    i_max: tl.constexpr,
    N_TOTAL: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N_TOTAL:
        return

    pid_b = pid // (N_TOTAL // tl.num_programs(0)) # simplified batch index

    i_ff = tl.load(i_feedforward_ptr + pid).to(tl.float32)
    i_rec = tl.load(i_recurrent_ptr + pid).to(tl.float32)
    ne = tl.load(ne_ptr + 0).to(tl.float32)
    ach = tl.load(ach_ptr + 0).to(tl.float32)

    # Multiplicative NE gain scaling of feedforward drive
    gain_ne = gamma_base + gamma_ne * ne
    # ACh suppression of intracortical recurrent noise
    gain_rec = 1.0 - beta_ach * ach
    if gain_rec < 0.1:
        gain_rec = 0.1

    i_net = (gain_ne * i_ff) + (gain_rec * i_rec)

    # Saturation clamp
    if i_net > i_max:
        i_net = i_max
    if i_net < -i_max:
        i_net = -i_max

    tl.store(i_eff_ptr + pid, i_net.to(tl.float16))

def apply_neuromodulatory_gain_triton(
    i_feedforward: torch.Tensor,
    i_recurrent: torch.Tensor,
    i_eff: torch.Tensor,
    ne: torch.Tensor,
    ach: torch.Tensor,
    gamma_base: float = 1.0,
    gamma_ne: float = 1.8,
    beta_ach: float = 0.6,
    i_max: float = 50.0
) -> torch.Tensor:
    """Evaluates neuromodulatory gain scaling in GPU SRAM."""
    assert i_feedforward.is_cuda and i_recurrent.is_cuda and i_eff.is_cuda
    N_TOTAL = i_feedforward.numel()
    grid = (N_TOTAL,)
    _fused_neuromodulatory_gain_kernel[grid](
        i_feedforward, i_recurrent, i_eff, ne, ach,
        gamma_base=float(gamma_base),
        gamma_ne=float(gamma_ne),
        beta_ach=float(beta_ach),
        i_max=float(i_max),
        N_TOTAL=N_TOTAL
    )
    return i_eff

def compute_neuromodulatory_gain_step(
    i_ff: float,
    i_rec: float,
    ne: float,
    ach: float,
    gamma_base: float = 1.0,
    gamma_ne: float = 1.8,
    beta_ach: float = 0.6,
    i_max: float = 50.0
) -> float:
    """Analytical CPU evaluation of effective neuromodulated current."""
    gain_ne = gamma_base + gamma_ne * ne
    gain_rec = max(0.1, 1.0 - beta_ach * ach)
    i_net = (gain_ne * i_ff) + (gain_rec * i_rec)
    return max(-i_max, min(i_max, i_net))
