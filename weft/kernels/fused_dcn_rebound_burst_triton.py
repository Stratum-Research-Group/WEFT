"""
Cerebellar Deep Nuclei (DCN) Disinhibition & Rebound Bursting Kernel in Triton 3.9.

Models:
1. Tonic Purkinje Cell GABAergic Inhibition:
   I_PC = W_PC_to_DCN @ x_PC
   Maintains DCN neurons in hyperpolarized regime (-70 to -65 mV), de-inactivating
   low-threshold T-type Ca2+ conductances:
   dh_T / dt = (h_inf(V) - h_T) / tau_h
   where h_inf(V) approaches 1.0 at hyperpolarized potentials.
2. Purkinje Cell Pause / Disinhibition Event:
   When motor cortex / climbing fiber inputs pause Purkinje firing (x_PC -> 0),
   the sudden release of GABAergic inhibition allows the de-inactivated T-type Ca2+
   conductance and I_h pacemaker currents to trigger an explosive high-frequency
   rebound burst (150-250 Hz).
3. Downstream Motor Command Routing:
   Y_rubrospinal = W_rubro @ y_DCN
   Y_thalamus    = W_thalamus @ y_DCN
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
from enum import IntEnum


class DCNFiringMode(IntEnum):
    TONIC_SUPPRESSED = 0
    BASELINE_SPIKING = 1
    EXPLOSIVE_REBOUND = 2


def fused_dcn_rebound_burst(
    pc_rate: torch.Tensor,
    h_T: torch.Tensor,
    motor_threshold: float = 150.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates DCN post-inhibitory rebound burst dynamics and firing mode.

    Args:
        pc_rate: Purkinje input firing rate [N] (Hz)
        h_T: T-type CaV3.1 de-inactivation gate [N] [0, 1]
        motor_threshold: Minimum rate (Hz) defining effective motor command

    Returns:
        (burst_rate, mode)
    """
    is_paused = pc_rate < 10.0
    is_primed = h_T >= 0.70
    is_suppressed = pc_rate > 50.0

    burst_rate = torch.empty_like(pc_rate)
    mode = torch.empty(pc_rate.shape, dtype=torch.long, device=pc_rate.device)

    # Rebound burst condition: Purkinje pause + primed de-inactivation gate
    rebound_mask = is_paused & is_primed
    suppress_mask = is_suppressed & (~rebound_mask)
    baseline_mask = (~rebound_mask) & (~suppress_mask)

    burst_rate[rebound_mask] = motor_threshold + 70.0
    mode[rebound_mask] = DCNFiringMode.EXPLOSIVE_REBOUND

    burst_rate[suppress_mask] = 12.0
    mode[suppress_mask] = DCNFiringMode.TONIC_SUPPRESSED

    burst_rate[baseline_mask] = 45.0
    mode[baseline_mask] = DCNFiringMode.BASELINE_SPIKING

    return burst_rate, mode


@triton.jit
def _dcn_rebound_step_kernel(
    V_IN_ptr,          # [B, N_DCN] - membrane potential V
    H_T_ptr,           # [B, N_DCN] - T-type Ca2+ de-inactivation state h_T
    X_PC_ptr,          # [B, N_PC]  - Purkinje cell firing rates
    W_PC_ptr,          # [N_DCN, N_PC] - Purkinje-to-DCN inhibitory weights
    I_EXC_ptr,         # [B, N_DCN] - mossy fiber / climbing fiber collateral excitation
    V_OUT_ptr,         # [B, N_DCN] - updated membrane potential
    H_T_OUT_ptr,       # [B, N_DCN] - updated h_T state
    Y_BURST_ptr,       # [B, N_DCN] - output rebound burst rate (Hz)
    g_T: float,        # T-type Ca2+ conductance
    E_Ca: float,       # Calcium reversal potential (+120 mV)
    dt: float,         # Integration step
    B: tl.constexpr,
    N_DCN: tl.constexpr,
    N_PC: tl.constexpr,
    BLOCK_DCN: tl.constexpr,
    BLOCK_PC: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)

    offs_d = pid_d * BLOCK_DCN + tl.arange(0, BLOCK_DCN)
    mask_d = offs_d < N_DCN

    # Accumulate Purkinje GABAergic inhibition
    acc_pc = tl.zeros([BLOCK_DCN], dtype=tl.float32)

    for k in range(0, N_PC, BLOCK_PC):
        offs_k = k + tl.arange(0, BLOCK_PC)
        mask_k = offs_k < N_PC

        x_pc = tl.load(X_PC_ptr + pid_b * N_PC + offs_k, mask=mask_k, other=0.0)
        w_block = tl.load(
            W_PC_ptr + offs_d[:, None] * N_PC + offs_k[None, :],
            mask=(mask_d[:, None] & mask_k[None, :]),
            other=0.0
        )
        acc_pc += tl.sum(w_block * x_pc[None, :], axis=1)

    # Current state
    v_curr = tl.load(V_IN_ptr + pid_b * N_DCN + offs_d, mask=mask_d, other=-65.0)
    h_curr = tl.load(H_T_ptr + pid_b * N_DCN + offs_d, mask=mask_d, other=0.8)
    i_exc = tl.load(I_EXC_ptr + pid_b * N_DCN + offs_d, mask=mask_d, other=1.0)

    # De-inactivation kinetics: hyperpolarization (v < -65) increases h_T toward 1.0
    # h_inf = 1.0 / (1.0 + exp((V + 70) / 4))
    h_target = tl.where(v_curr < -60.0, 1.0, 0.0)
    h_next = h_curr + dt * (h_target - h_curr) * 0.2

    # T-type current: I_T = g_T * h_T * (E_Ca - V) when depolarizing from negative
    # Rebound drive is active when Purkinje inhibition drops (acc_pc < 0.2) and h_curr is high
    is_disinhibited = acc_pc < 0.2
    i_T = tl.where(is_disinhibited, g_T * h_next * (E_Ca - v_curr) * 0.05, 0.0)

    # Voltage evolution
    v_next = v_curr + dt * (- (v_curr - (-65.0)) - acc_pc * 20.0 + i_exc + i_T)

    # Rebound burst rate (Hz): scaling above threshold -50 mV
    burst_rate = tl.maximum((v_next - (-50.0)) * 12.0, 0.0)

    tl.store(V_OUT_ptr + pid_b * N_DCN + offs_d, v_next, mask=mask_d)
    tl.store(H_T_OUT_ptr + pid_b * N_DCN + offs_d, h_next, mask=mask_d)
    tl.store(Y_BURST_ptr + pid_b * N_DCN + offs_d, burst_rate, mask=mask_d)


def dcn_rebound_burst_step(
    v_curr: torch.Tensor,
    h_curr: torch.Tensor,
    x_pc: torch.Tensor,
    w_pc_to_dcn: torch.Tensor,
    i_exc: torch.Tensor,
    g_T: float = 2.0,
    dt: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates DCN post-inhibitory rebound burst generation upon Purkinje pause.

    Args:
        v_curr:      [B, N_DCN] current DCN membrane potential
        h_curr:      [B, N_DCN] T-type Ca2+ de-inactivation state
        x_pc:        [B, N_PC]  Purkinje cell firing rates (0 during pause)
        w_pc_to_dcn: [N_DCN, N_PC] Purkinje-to-DCN synaptic weights
        i_exc:       [B, N_DCN] collateral excitation
        g_T:         T-type conductance
        dt:          time step

    Returns:
        (v_next, h_next, burst_rate_hz)
    """
    B, N_PC = x_pc.shape
    N_DCN = w_pc_to_dcn.shape[0]
    device = v_curr.device

    if v_curr.is_cuda:
        v_next = torch.empty((B, N_DCN), device=device, dtype=torch.float32)
        h_next = torch.empty((B, N_DCN), device=device, dtype=torch.float32)
        burst_rate = torch.empty((B, N_DCN), device=device, dtype=torch.float32)

        BLOCK_DCN = 128 if N_DCN >= 128 else 32
        BLOCK_PC = 64 if N_PC >= 64 else 16
        grid = (B, triton.cdiv(N_DCN, BLOCK_DCN))

        _dcn_rebound_step_kernel[grid](
            v_curr.contiguous(),
            h_curr.contiguous(),
            x_pc.contiguous(),
            w_pc_to_dcn.contiguous(),
            i_exc.contiguous(),
            v_next,
            h_next,
            burst_rate,
            float(g_T),
            120.0,
            float(dt),
            B=B,
            N_DCN=N_DCN,
            N_PC=N_PC,
            BLOCK_DCN=BLOCK_DCN,
            BLOCK_PC=BLOCK_PC,
        )
        return v_next, h_next, burst_rate
    else:
        # High-performance PyTorch CPU fallback
        gaba_pc = torch.matmul(x_pc, w_pc_to_dcn.t())
        h_target = torch.where(v_curr < -60.0, torch.ones_like(h_curr), torch.zeros_like(h_curr))
        h_next = h_curr + dt * (h_target - h_curr) * 0.2

        is_disinhibited = (gaba_pc < 0.2).float()
        i_T = is_disinhibited * g_T * h_next * (120.0 - v_curr) * 0.05

        v_next = v_curr + dt * (- (v_curr - (-65.0)) - gaba_pc * 20.0 + i_exc + i_T)
        burst_rate = torch.relu((v_next - (-50.0)) * 12.0)

        return v_next, h_next, burst_rate
