"""
Pedunculopontine Tegmental Nucleus (PPN) Locomotor Gating & Stepping Rhythm Kernel in Triton 3.9.

Models:
1. Basal Ganglia Inhibitory Brake:
   Tonic GABAergic inhibition from GPi/SNr: I_GABA
   Basal ganglia disinhibition: Net PPN drive = ReLU( I_stim - W_gpi @ x_gpi )
2. PPN Neuronal Subpopulations:
   - PPN Glutamatergic (PPN-Glu): Initiates and sustains locomotion via Reticulospinal tract (MLR command).
   - PPN Cholinergic (PPN-ACh): Modulates muscle tone, postural tone, and rhythmic gait transitions.
3. Reticulospinal Transmission & Central Pattern Generator (CPG) Frequency:
   f_CPG = alpha_step * ReLU( y_glu - theta_locomotor )
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _ppn_locomotor_step_kernel(
    X_GPI_ptr,          # [B, N_GPI] - GPi/SNr GABAergic brake rates
    W_GPI_ptr,          # [N_PPN, N_GPI] - descending inhibitory weights
    I_DRIVE_ptr,        # [B, N_PPN] - excitatory drive (cortical / limbic inputs)
    Y_GLU_ptr,          # [B, N_PPN] - output PPN glutamatergic firing rate
    Y_ACH_ptr,          # [B, N_PPN] - output PPN cholinergic firing rate
    STEP_FREQ_ptr,      # [B] - spinal CPG stepping frequency (Hz)
    theta_loco: float,  # threshold for locomotor activation
    alpha_step: float,  # stepping frequency gain
    ach_gain: float,    # cholinergic modulation gain
    B: tl.constexpr,
    N_PPN: tl.constexpr,
    N_GPI: tl.constexpr,
    BLOCK_PPN: tl.constexpr,
    BLOCK_GPI: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_p = tl.program_id(axis=1)

    offs_p = pid_p * BLOCK_PPN + tl.arange(0, BLOCK_PPN)
    mask_p = offs_p < N_PPN

    # Accumulate tonic GABAergic brake from GPi/SNr
    acc_gaba = tl.zeros([BLOCK_PPN], dtype=tl.float32)
    for k in range(0, N_GPI, BLOCK_GPI):
        offs_k = k + tl.arange(0, BLOCK_GPI)
        mask_k = offs_k < N_GPI

        x_gpi = tl.load(X_GPI_ptr + pid_b * N_GPI + offs_k, mask=mask_k, other=0.0)
        w_block = tl.load(
            W_GPI_ptr + offs_p[:, None] * N_GPI + offs_k[None, :],
            mask=(mask_p[:, None] & mask_k[None, :]),
            other=0.0
        )
        acc_gaba += tl.sum(w_block * x_gpi[None, :], axis=1)

    # Load intrinsic excitatory drive
    i_drive = tl.load(I_DRIVE_ptr + pid_b * N_PPN + offs_p, mask=mask_p, other=0.0)

    # Net PPN disinhibition
    net_drive = tl.maximum(i_drive - acc_gaba, 0.0)

    # Glutamatergic output: direct projection to reticulospinal neurons
    y_glu = net_drive

    # Cholinergic output: neuromodulatory tone
    y_ach = tl.maximum(net_drive * ach_gain, 0.0)

    tl.store(Y_GLU_ptr + pid_b * N_PPN + offs_p, y_glu, mask=mask_p)
    tl.store(Y_ACH_ptr + pid_b * N_PPN + offs_p, y_ach, mask=mask_p)


def ppn_locomotor_gating(
    x_gpi: torch.Tensor,
    w_gpi_to_ppn: torch.Tensor,
    i_drive: torch.Tensor,
    theta_loco: float = 0.2,
    alpha_step: float = 0.08,
    ach_gain: float = 0.85
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates PPN disinhibition, cholinergic/glutamatergic activation, and CPG frequency scaling.

    Args:
        x_gpi:        [B, N_GPI] GPi/SNr tonic GABAergic inhibition rate
        w_gpi_to_ppn: [N_PPN, N_GPI] GPi to PPN inhibitory synaptic weights
        i_drive:      [B, N_PPN] descending cortical/limbic drive
        theta_loco:   locomotor initiation threshold
        alpha_step:   scaling factor from PPN-Glu rate to spinal CPG frequency
        ach_gain:     gain factor for PPN-ACh neuromodulatory output

    Returns:
        (y_glu, y_ach, cpg_step_freq_hz)
    """
    B, N_GPI = x_gpi.shape
    N_PPN = w_gpi_to_ppn.shape[0]
    device = x_gpi.device

    if x_gpi.is_cuda:
        y_glu = torch.empty((B, N_PPN), device=device, dtype=torch.float32)
        y_ach = torch.empty((B, N_PPN), device=device, dtype=torch.float32)
        step_freq = torch.zeros(B, device=device, dtype=torch.float32)

        BLOCK_PPN = 128 if N_PPN >= 128 else 32
        BLOCK_GPI = 64 if N_GPI >= 64 else 16
        grid = (B, triton.cdiv(N_PPN, BLOCK_PPN))

        _ppn_locomotor_step_kernel[grid](
            x_gpi.contiguous(),
            w_gpi_to_ppn.contiguous(),
            i_drive.contiguous(),
            y_glu,
            y_ach,
            step_freq,
            float(theta_loco),
            float(alpha_step),
            float(ach_gain),
            B=B,
            N_PPN=N_PPN,
            N_GPI=N_GPI,
            BLOCK_PPN=BLOCK_PPN,
            BLOCK_GPI=BLOCK_GPI,
        )
        # Compute mean spinal CPG stepping frequency
        glu_excess = torch.relu(y_glu - theta_loco)
        cpg_freq = alpha_step * torch.mean(glu_excess, dim=-1)
        return y_glu, y_ach, cpg_freq
    else:
        # High-performance PyTorch CPU fallback
        gaba_inhibition = torch.matmul(x_gpi, w_gpi_to_ppn.t())
        net_drive = torch.relu(i_drive - gaba_inhibition)

        y_glu = net_drive
        y_ach = torch.relu(net_drive * ach_gain)

        glu_excess = torch.relu(y_glu - theta_loco)
        cpg_freq = alpha_step * torch.mean(glu_excess, dim=-1)

        return y_glu, y_ach, cpg_freq
