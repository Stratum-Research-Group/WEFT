"""
Subicular Sharp-Wave Ripple (SWR) Bursting & Replay Broadcasting Kernel in Triton 3.9.

Models:
1. Subicular Pyramidal Cell Bimodal Firing:
   - Regular spiking mode (sensory passthrough): V_m driven by CA1 synaptic drive.
   - Burst-firing mode (SWR replay): Triggered by strong dendritic depolarizing plateau
     and low-threshold T-type Ca2+ channel activation (I_T = g_T * m_T^2 * h_T * (V - E_Ca)).
2. Sharp-Wave Ripple Generation:
   - High-frequency wave-packet modulation (150-250 Hz sinusoidal packet) gating burst discharges.
3. Broadcasting Fan-Out Routing:
   - Concurrent parallel projections:
     Y_mPFC = W_mPFC @ y_burst
     Y_EC_deep = W_EC @ y_burst
     Y_MB = W_MB @ y_burst
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _subiculum_swr_burst_kernel(
    V_IN_ptr,          # [B, N] - input membrane potential / CA1 drive
    BURST_OUT_ptr,     # [B, N] - output burst activity
    SWR_ENVELOPE_ptr,  # [B]    - SWR ripple amplitude factor
    g_T: float,        # T-type Ca2+ conductance
    E_Ca: float,       # Reversal potential for Ca2+
    v_thresh: float,   # Burst firing threshold
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    # Load SWR amplitude envelope
    swr_amp = tl.load(SWR_ENVELOPE_ptr + pid_b)

    # Load input voltage
    v_in = tl.load(V_IN_ptr + pid_b * N + offs, mask=mask, other=-70.0)

    # T-type Ca2+ activation approximation: m_inf = 1.0 / (1.0 + exp(-(v - v_half) / k))
    # For subicular burst firing, v > -55 mV initiates low-threshold burst
    v_diff = v_in - v_thresh
    is_bursting = v_diff > 0.0

    # Low-threshold calcium current drive
    i_ca = tl.where(is_bursting, g_T * (E_Ca - v_in), 0.0)

    # Burst envelope gated by SWR ripple oscillatory packet
    burst_activity = tl.maximum(v_diff + i_ca * 0.01, 0.0) * swr_amp

    # Store burst activity
    tl.store(BURST_OUT_ptr + pid_b * N + offs, burst_activity, mask=mask)


def subiculum_replay_broadcast(
    v_ca1_drive: torch.Tensor,
    w_mpfc: torch.Tensor,
    w_ec_deep: torch.Tensor,
    swr_active: bool = True,
    ripple_freq_hz: float = 200.0,
    time_ms: float = 50.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates subicular burst generation during SWR and broadcasts to neocortical targets.

    Args:
        v_ca1_drive: [B, N] depolarizing drive from CA1
        w_mpfc:      [N_mPFC, N] projection weights to medial prefrontal cortex
        w_ec_deep:   [N_EC, N]   projection weights to deep entorhinal layers
        swr_active:  whether network is currently in Sharp-Wave Ripple state
        ripple_freq_hz: frequency of ripple modulation
        time_ms: current elapsed time in SWR event

    Returns:
        (y_mpfc, y_ec_deep, burst_activity)
    """
    B, N = v_ca1_drive.shape
    device = v_ca1_drive.device

    # Calculate SWR ripple envelope: 1.0 + 0.8 * cos(2 * pi * f * t)
    if swr_active:
        t_sec = time_ms / 1000.0
        swr_val = float(1.0 + 0.75 * torch.cos(torch.tensor(2.0 * 3.14159265 * ripple_freq_hz * t_sec)).item())
    else:
        swr_val = 0.25 # Quiescent / passthrough baseline

    if v_ca1_drive.is_cuda:
        burst_activity = torch.empty((B, N), device=device, dtype=torch.float32)
        swr_env = torch.tensor([swr_val] * B, device=device, dtype=torch.float32)
        BLOCK_N = 128 if N >= 128 else 32
        grid = (B, triton.cdiv(N, BLOCK_N))

        _subiculum_swr_burst_kernel[grid](
            v_ca1_drive.contiguous(),
            burst_activity,
            swr_env,
            g_T=1.5,
            E_Ca=120.0,
            v_thresh=-55.0,
            B=B,
            N=N,
            BLOCK_N=BLOCK_N,
        )
    else:
        # High-performance PyTorch CPU fallback
        v_diff = torch.relu(v_ca1_drive - (-55.0))
        i_ca = torch.where(v_diff > 0, 1.5 * (120.0 - v_ca1_drive), torch.zeros_like(v_ca1_drive))
        burst_activity = (v_diff + i_ca * 0.01) * swr_val

    # Broadcast to targets via matrix multiplication
    y_mpfc = torch.relu(torch.matmul(burst_activity, w_mpfc.t()))
    y_ec_deep = torch.relu(torch.matmul(burst_activity, w_ec_deep.t()))

    return y_mpfc, y_ec_deep, burst_activity
