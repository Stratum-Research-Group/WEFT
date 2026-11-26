"""
WEFT Fused Cerebellar Long-Term Depression (LTD) Kernel (Triton 3.9).
Milestone Paper #148 Series.
Models heterosynaptic coincidence of Parallel Fiber (PF) glutamate and
Climbing Fiber (CF) complex spike depolarization:
Delta_W = -eta_LTD * (PF_active * CF_error) + eta_LTP * (PF_active * (1 - CF_error))
operating with bounded synaptic saturation directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_cerebellar_ltd_kernel(
    pf_activity_ptr, # [B, N_PF] parallel fiber activity in [0, 1]
    cf_error_ptr,    # [B] climbing fiber binary/graded error in [0, 1]
    weights_ptr,     # [B, N_PF] PF -> Purkinje synaptic weights
    eta_ltd: tl.constexpr,   # LTD learning rate e.g. 0.05
    eta_ltp: tl.constexpr,   # Slow LTP recovery rate e.g. 0.005
    w_min: tl.constexpr,     # minimal weight e.g. 0.05
    w_max: tl.constexpr,     # maximal weight e.g. 1.0
    N_PF: tl.constexpr
):
    pid_pf = tl.program_id(0)
    pid_b = tl.program_id(1)

    if pid_pf >= N_PF:
        return

    offset = pid_b * N_PF + pid_pf
    pf = tl.load(pf_activity_ptr + offset).to(tl.float32)
    cf = tl.load(cf_error_ptr + pid_b).to(tl.float32)
    w_curr = tl.load(weights_ptr + offset).to(tl.float32)

    # Coincidence rule:
    # If CF is active, induce LTD (decrease weight).
    # If CF is inactive, allow slow baseline LTP recovery.
    dw = 0.0
    if cf > 0.5:
        dw = -eta_ltd * pf
    else:
        dw = eta_ltp * pf

    w_new = w_curr + dw
    # Synaptic bounds clamping
    if w_new < w_min:
        w_new = w_min
    if w_new > w_max:
        w_new = w_max

    tl.store(weights_ptr + offset, w_new.to(tl.float16))

def apply_cerebellar_ltd_triton(
    pf_activity: torch.Tensor,
    cf_error: torch.Tensor,
    weights: torch.Tensor,
    eta_ltd: float = 0.05,
    eta_ltp: float = 0.005,
    w_min: float = 0.05,
    w_max: float = 1.0
) -> torch.Tensor:
    """Applies heterosynaptic cerebellar LTD updates in GPU SRAM."""
    assert pf_activity.is_cuda and cf_error.is_cuda and weights.is_cuda
    B, N_PF = pf_activity.shape
    grid = (N_PF, B)
    _fused_cerebellar_ltd_kernel[grid](
        pf_activity, cf_error, weights,
        eta_ltd=float(eta_ltd),
        eta_ltp=float(eta_ltp),
        w_min=float(w_min),
        w_max=float(w_max),
        N_PF=N_PF
    )
    return weights

def compute_cerebellar_ltd_step(
    w_curr: float,
    pf: float,
    cf: float,
    eta_ltd: float = 0.05,
    eta_ltp: float = 0.005,
    w_min: float = 0.05,
    w_max: float = 1.0
) -> float:
    """Analytical CPU reference evaluation of cerebellar LTD step."""
    if cf > 0.5:
        dw = -eta_ltd * pf
    else:
        dw = eta_ltp * pf
    return max(w_min, min(w_max, w_curr + dw))
