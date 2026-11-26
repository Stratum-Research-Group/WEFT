"""
WEFT Fused Three-Factor Dopaminergic STDP Kernel (Triton 3.9).
Milestone Paper #145 Series.
Computes eligibility trace decay e(t) and phasic dopamine modulation delta_DA:
Delta_W = eta * delta_DA * e(t)
directly in GPU SRAM registers across recurrent cortical-striatal synaptic networks.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_three_factor_stdp_kernel(
    traces_ptr,      # [B, N_pre, N_post] eligibility traces e(t)
    weights_ptr,     # [B, N_pre, N_post] current synaptic weights
    dopamine_ptr,    # [B] phasic dopamine scalar delta_DA per batch
    eta: tl.constexpr,       # learning rate
    w_min: tl.constexpr,     # lower bound
    w_max: tl.constexpr,     # upper bound
    trace_decay: tl.constexpr, # decay factor per step e.g. exp(-dt / tau_e)
    N_PRE: tl.constexpr,
    N_POST: tl.constexpr
):
    pid_pre = tl.program_id(0)
    pid_post = tl.program_id(1)
    pid_b = tl.program_id(2)

    offset = pid_b * (N_PRE * N_POST) + pid_pre * N_POST + pid_post
    e_curr = tl.load(traces_ptr + offset).to(tl.float32)
    w_curr = tl.load(weights_ptr + offset).to(tl.float32)
    da = tl.load(dopamine_ptr + pid_b).to(tl.float32)

    # Three-factor update: Delta_W = eta * da * e_curr
    dw = eta * da * e_curr
    w_new = w_curr + dw

    # Saturation clamp in [w_min, w_max]
    if w_new < w_min:
        w_new = w_min
    if w_new > w_max:
        w_new = w_max

    # Update eligibility trace decay
    e_new = e_curr * trace_decay

    tl.store(weights_ptr + offset, w_new.to(tl.float16))
    tl.store(traces_ptr + offset, e_new.to(tl.float16))

def apply_three_factor_stdp_triton(
    traces: torch.Tensor,
    weights: torch.Tensor,
    dopamine: torch.Tensor,
    eta: float = 0.01,
    w_min: float = 0.0,
    w_max: float = 1.0,
    trace_decay: float = 0.95
) -> torch.Tensor:
    """Applies in-place three-factor dopamine STDP update on CUDA GPU."""
    assert traces.is_cuda and weights.is_cuda and dopamine.is_cuda
    B, N_pre, N_post = weights.shape
    grid = (N_pre, N_post, B)
    _fused_three_factor_stdp_kernel[grid](
        traces, weights, dopamine,
        eta=float(eta),
        w_min=float(w_min),
        w_max=float(w_max),
        trace_decay=float(trace_decay),
        N_PRE=N_pre,
        N_POST=N_post
    )
    return weights

def compute_three_factor_step(
    w_curr: float,
    e_curr: float,
    da: float,
    eta: float = 0.01,
    trace_decay: float = 0.95,
    w_min: float = 0.0,
    w_max: float = 1.0
) -> tuple[float, float]:
    """Analytical CPU evaluation of three-factor dopamine step."""
    dw = eta * da * e_curr
    w_new = max(w_min, min(w_max, w_curr + dw))
    e_new = e_curr * trace_decay
    return w_new, e_new
