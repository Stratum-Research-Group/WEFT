"""
WEFT Fused Amygdalar Associative Plasticity & Threat Gating Kernel (Triton 3.9).
Milestone Paper #153 Series.
Models:
1. Fast CS-US associative Hebbian LTP at BLA synapses:
   W_cs[t+1] = clamp(W_cs + eta_ltp * CS * US, w_min, w_max)
2. Infralimbic prefrontal extinction gating via Intercalated (ITC) GABAergic cells:
   I_cea = max(0.0, W_cs * CS - alpha_itc * vmPFC_extinction)
directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_amygdalar_plasticity_kernel(
    w_cs_ptr,      # [B, N] thalamo/cortico-BLA synaptic weights
    cs_drive_ptr,  # [B, N] conditioned stimulus inputs
    us_drive_ptr,  # [B] unconditioned stimulus / shock teacher signal
    vmpfc_ptr,     # [B] vmPFC extinction drive in [0, 1]
    i_cea_ptr,     # [B, N] output Central Nucleus (CeA) threat output
    eta_ltp: tl.constexpr,   # fast associative LTP learning rate e.g. 0.15
    alpha_itc: tl.constexpr, # ITC inhibitory gain e.g. 0.95
    w_min: tl.constexpr,     # minimal weight e.g. 0.1
    w_max: tl.constexpr,     # maximal weight e.g. 1.0
    N_TOTAL: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N_TOTAL:
        return

    w = tl.load(w_cs_ptr + pid).to(tl.float32)
    cs = tl.load(cs_drive_ptr + pid).to(tl.float32)
    us = tl.load(us_drive_ptr + 0).to(tl.float32)
    vmpfc = tl.load(vmpfc_ptr + 0).to(tl.float32)

    # 1. Associative Hebbian LTP: CS-US coincidence
    dw = eta_ltp * cs * us
    w_new = w + dw
    if w_new < w_min:
        w_new = w_min
    if w_new > w_max:
        w_new = w_max

    # 2. BLA excitatory output -> CeA threat gate
    bla_out = w_new * cs

    # 3. ITC GABAergic clamp driven by vmPFC
    itc_clamp = alpha_itc * vmpfc
    cea_out = bla_out - itc_clamp
    if cea_out < 0.0:
        cea_out = 0.0

    tl.store(w_cs_ptr + pid, w_new.to(tl.float16))
    tl.store(i_cea_ptr + pid, cea_out.to(tl.float16))

def apply_amygdalar_plasticity_triton(
    w_cs: torch.Tensor,
    cs_drive: torch.Tensor,
    us_drive: torch.Tensor,
    vmpfc: torch.Tensor,
    i_cea: torch.Tensor,
    eta_ltp: float = 0.15,
    alpha_itc: float = 0.95,
    w_min: float = 0.1,
    w_max: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies amygdalar associative plasticity and threat gating in GPU SRAM."""
    assert w_cs.is_cuda and cs_drive.is_cuda and i_cea.is_cuda
    N_TOTAL = w_cs.numel()
    grid = (N_TOTAL,)
    _fused_amygdalar_plasticity_kernel[grid](
        w_cs, cs_drive, us_drive, vmpfc, i_cea,
        eta_ltp=float(eta_ltp),
        alpha_itc=float(alpha_itc),
        w_min=float(w_min),
        w_max=float(w_max),
        N_TOTAL=N_TOTAL
    )
    return w_cs, i_cea

def compute_amygdalar_step(
    w_curr: float,
    cs: float,
    us: float,
    vmpfc: float,
    eta_ltp: float = 0.15,
    alpha_itc: float = 0.95,
    w_min: float = 0.1,
    w_max: float = 1.0
) -> tuple[float, float]:
    """Analytical CPU evaluation of BLA weight and CeA output."""
    w_new = max(w_min, min(w_max, w_curr + eta_ltp * cs * us))
    bla_out = w_new * cs
    cea_out = max(0.0, bla_out - alpha_itc * vmpfc)
    return w_new, cea_out
