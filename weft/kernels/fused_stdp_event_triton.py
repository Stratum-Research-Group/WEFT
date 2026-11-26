"""
WEFT Fused Spike-Timing-Dependent Plasticity (STDP) Event Triton 3.9 Kernel.
Milestone Paper #144.
Computes event-driven synaptic weight updates based on pre- and post-synaptic spike arrival times:
Delta_W = A_pos * exp(-dt / tau_pos) if dt > 0 else -A_neg * exp(dt / tau_neg)
directly in GPU SRAM registers across large recurrent cortical synaptic matrices.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_stdp_event_kernel(
    pre_spikes_ptr,  # [B, N_pre] spike timestamps (ms)
    post_spikes_ptr, # [B, N_post] spike timestamps (ms)
    weights_ptr,     # [B, N_pre, N_post] current synaptic weights
    w_min: tl.constexpr,
    w_max: tl.constexpr,
    tau_pos: tl.constexpr, # e.g. 20.0 ms
    tau_neg: tl.constexpr, # e.g. 20.0 ms
    a_pos: tl.constexpr,   # e.g. 0.05
    a_neg: tl.constexpr,   # e.g. 0.06
    N_PRE: tl.constexpr,
    N_POST: tl.constexpr
):
    pid_pre = tl.program_id(0)
    pid_post = tl.program_id(1)
    pid_b = tl.program_id(2)

    t_pre = tl.load(pre_spikes_ptr + pid_b * N_PRE + pid_pre).to(tl.float32)
    t_post = tl.load(post_spikes_ptr + pid_b * N_POST + pid_post).to(tl.float32)

    offset_w = pid_b * (N_PRE * N_POST) + pid_pre * N_POST + pid_post
    w_curr = tl.load(weights_ptr + offset_w).to(tl.float32)

    dt = t_post - t_pre # Pre before post is positive

    # STDP Exponential update
    dw = 0.0
    if dt > 0.0:
        dw = a_pos * tl.exp(-dt / tau_pos)
    elif dt < 0.0:
        dw = -a_neg * tl.exp(dt / tau_neg)

    w_new = w_curr + dw
    # Synaptic homeostatic clamping in [w_min, w_max]
    if w_new < w_min:
        w_new = w_min
    if w_new > w_max:
        w_new = w_max

    tl.store(weights_ptr + offset_w, w_new.to(tl.float16))

def apply_stdp_updates_triton(
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    weights: torch.Tensor,
    w_min: float = 0.0,
    w_max: float = 1.0,
    tau_pos: float = 20.0,
    tau_neg: float = 20.0,
    a_pos: float = 0.05,
    a_neg: float = 0.06
) -> torch.Tensor:
    """
    Applies event-driven STDP weight updates in-place on CUDA GPU.
    pre_spikes: [B, N_pre]
    post_spikes: [B, N_post]
    weights: [B, N_pre, N_post] (float16)
    """
    assert pre_spikes.is_cuda and post_spikes.is_cuda and weights.is_cuda
    B, N_pre = pre_spikes.shape
    _, N_post = post_spikes.shape

    grid = (N_pre, N_post, B)

    _fused_stdp_event_kernel[grid](
        pre_spikes.contiguous(),
        post_spikes.contiguous(),
        weights,
        w_min=float(w_min),
        w_max=float(w_max),
        tau_pos=float(tau_pos),
        tau_neg=float(tau_neg),
        a_pos=float(a_pos),
        a_neg=float(a_neg),
        N_PRE=N_pre,
        N_POST=N_post
    )
    return weights

def compute_stdp_delta(
    dt: float,
    a_plus: float = 0.05,
    a_minus: float = 0.06,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0
) -> float:
    """CPU analytical calculation of delta W given dt = t_post - t_pre."""
    import math
    if dt > 0.0:
        return a_plus * math.exp(-dt / tau_plus)
    elif dt < 0.0:
        return -a_minus * math.exp(dt / tau_minus)
    return 0.0

def apply_stdp_event_driven(
    w_curr: float,
    dt: float,
    a_plus: float = 0.05,
    a_minus: float = 0.06,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    w_min: float = 0.0,
    w_max: float = 1.0
) -> float:
    """Applies STDP update with bounded interval [w_min, w_max] clamping."""
    dw = compute_stdp_delta(dt, a_plus, a_minus, tau_plus, tau_minus)
    w_new = w_curr + dw
    return max(w_min, min(w_max, w_new))

