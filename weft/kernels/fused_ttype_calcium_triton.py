"""
WEFT Fused T-Type Calcium Channel Dynamics Kernel (Triton 3.9).
Milestone Paper #146 Series.
Models voltage-dependent T-type calcium channel (CaV3.1) activation m_T and
inactivation h_T gating dynamics during hyperpolarization-rebound LTS bursts:
I_T = g_T * (m_T)^2 * h_T * (V - E_Ca)
directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_ttype_calcium_kernel(
    v_ptr,       # [B, N] membrane potential (mV)
    h_ptr,       # [B, N] T-channel inactivation gating variable h_T in [0, 1]
    i_t_ptr,     # [B, N] output T-type calcium current (pA / pF)
    g_t: tl.constexpr,   # maximal conductance e.g. 2.0 mS/cm^2
    e_ca: tl.constexpr,  # reversal potential e.g. 120.0 mV
    dt: tl.constexpr,    # timestep (ms) e.g. 0.5
    N_TOTAL: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N_TOTAL:
        return

    v = tl.load(v_ptr + pid).to(tl.float32)
    h = tl.load(h_ptr + pid).to(tl.float32)

    # Steady-state activation: m_inf = 1 / (1 + exp(-(v + 57) / 6.2))
    m_inf = 1.0 / (1.0 + tl.exp(-(v + 57.0) / 6.2))

    # Steady-state inactivation: h_inf = 1 / (1 + exp((v + 81) / 4.0))
    # Hyperpolarization (v < -65) increases h_inf toward 1.0 (de-inactivation)
    h_inf = 1.0 / (1.0 + tl.exp((v + 81.0) / 4.0))

    # Inactivation time constant tau_h(V) (approx 20-50 ms)
    tau_h = 30.0 + 20.0 / (1.0 + tl.exp((v + 65.0) / 3.0))

    # Update h gating variable
    dh = (h_inf - h) * (dt / tau_h)
    h_new = h + dh
    if h_new < 0.0:
        h_new = 0.0
    if h_new > 1.0:
        h_new = 1.0

    # Low-threshold calcium current: I_T = g_T * (m_inf^2) * h_new * (v - e_ca)
    current = g_t * (m_inf * m_inf) * h_new * (v - e_ca)

    tl.store(h_ptr + pid, h_new.to(tl.float16))
    tl.store(i_t_ptr + pid, current.to(tl.float16))

def apply_ttype_calcium_triton(
    v: torch.Tensor,
    h: torch.Tensor,
    i_t: torch.Tensor,
    g_t: float = 2.0,
    e_ca: float = 120.0,
    dt: float = 0.5
) -> torch.Tensor:
    """Evaluates T-type calcium channel gating in GPU SRAM."""
    assert v.is_cuda and h.is_cuda and i_t.is_cuda
    N_TOTAL = v.numel()
    grid = (N_TOTAL,)
    _fused_ttype_calcium_kernel[grid](
        v, h, i_t,
        g_t=float(g_t),
        e_ca=float(e_ca),
        dt=float(dt),
        N_TOTAL=N_TOTAL
    )
    return i_t

def compute_ttype_calcium_step(
    v: float,
    h: float,
    g_t: float = 2.0,
    e_ca: float = 120.0,
    dt: float = 0.5
) -> tuple[float, float]:
    """Analytical CPU reference evaluation of T-channel step."""
    m_inf = 1.0 / (1.0 + math.exp(-(v + 57.0) / 6.2))
    h_inf = 1.0 / (1.0 + math.exp((v + 81.0) / 4.0))
    tau_h = 30.0 + 20.0 / (1.0 + math.exp((v + 65.0) / 3.0))
    dh = (h_inf - h) * (dt / tau_h)
    h_new = max(0.0, min(1.0, h + dh))
    i_t = g_t * (m_inf ** 2) * h_new * (v - e_ca)
    return h_new, i_t
