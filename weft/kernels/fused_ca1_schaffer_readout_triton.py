"""
Hippocampal CA1 Dual-Stream Comparator & Sequence Readout Kernel in Triton 3.9.

Models:
1. Schaffer Collateral input (CA3 -> CA1 proximal dendrite):
   I_CA3 = W_schaffer @ x_CA3
2. Temporoammonic direct path input (EC Layer III -> CA1 distal tuft):
   I_EC = W_ta @ x_EC
3. Theta-phase dependent gating:
   g_CA3(theta) = 0.5 * (1.0 - cos(theta))  # peak at trough (180 deg)
   g_EC(theta)  = 0.5 * (1.0 + cos(theta))  # peak at peak (0/360 deg)
4. Coincidence detection and mismatch computation:
   mismatch = || normalized(I_CA3) - normalized(I_EC) ||_2
   y_CA1 = relu(g_CA3 * I_CA3 + g_EC * I_EC - theta_inhibition)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _ca1_dual_stream_kernel(
    X_CA3_ptr,        # [B, N_CA3]
    X_EC_ptr,         # [B, N_EC]
    W_SCHAFFER_ptr,   # [N_CA1, N_CA3]
    W_TA_ptr,         # [N_CA1, N_EC]
    Y_CA1_ptr,        # [B, N_CA1]
    MISMATCH_ptr,     # [B]
    theta_rad: float, # Current theta phase in radians
    theta_inhibition: float,
    B: tl.constexpr,
    N_CA1: tl.constexpr,
    N_CA3: tl.constexpr,
    N_EC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CA1

    # Theta phase gating factors
    # cos(theta)
    cos_theta = tl.libdevice.cos(theta_rad)
    g_ca3 = 0.5 * (1.0 - cos_theta)
    g_ec  = 0.5 * (1.0 + cos_theta)

    # Accumulate Schaffer drive
    acc_ca3 = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k_start in range(0, N_CA3, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N_CA3
        x_ca3 = tl.load(X_CA3_ptr + pid_b * N_CA3 + offs_k, mask=mask_k, other=0.0)
        w_s = tl.load(
            W_SCHAFFER_ptr + offs_m[:, None] * N_CA3 + offs_k[None, :],
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0
        )
        acc_ca3 += tl.sum(w_s * x_ca3[None, :], axis=1)

    # Accumulate Temporoammonic drive
    acc_ec = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k_start in range(0, N_EC, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N_EC
        x_ec = tl.load(X_EC_ptr + pid_b * N_EC + offs_k, mask=mask_k, other=0.0)
        w_ta = tl.load(
            W_TA_ptr + offs_m[:, None] * N_EC + offs_k[None, :],
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0
        )
        acc_ec += tl.sum(w_ta * x_ec[None, :], axis=1)

    # Dual-stream integration
    net_drive = g_ca3 * acc_ca3 + g_ec * acc_ec - theta_inhibition
    y_ca1 = tl.maximum(net_drive, 0.0)

    # Store CA1 output
    tl.store(Y_CA1_ptr + pid_b * N_CA1 + offs_m, y_ca1, mask=mask_m)


def ca1_dual_stream_readout(
    x_ca3: torch.Tensor,
    x_ec: torch.Tensor,
    w_schaffer: torch.Tensor,
    w_ta: torch.Tensor,
    theta_rad: float = 3.14159265,
    theta_inhibition: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes CA1 dual-stream integration and mismatch detection.
    
    Args:
        x_ca3: [B, N_CA3] CA3 completed attractor pattern
        x_ec:  [B, N_EC]  EC Layer III sensory input pattern
        w_schaffer: [N_CA1, N_CA3] Schaffer collateral weight matrix
        w_ta:       [N_CA1, N_EC]  Temporoammonic direct weight matrix
        theta_rad: Theta oscillation phase (radians)
        theta_inhibition: Local interneuron inhibitory threshold
    
    Returns:
        (y_ca1, mismatch_score)
    """
    B, N_CA3 = x_ca3.shape
    _, N_EC = x_ec.shape
    N_CA1 = w_schaffer.shape[0]

    if x_ca3.is_cuda:
        y_ca1 = torch.empty((B, N_CA1), device=x_ca3.device, dtype=torch.float32)
        mismatch = torch.zeros(B, device=x_ca3.device, dtype=torch.float32)
        BLOCK_M = 128 if N_CA1 >= 128 else 32
        BLOCK_K = 128 if N_CA3 >= 128 else 32
        grid = (B, triton.cdiv(N_CA1, BLOCK_M))

        _ca1_dual_stream_kernel[grid](
            x_ca3.contiguous(),
            x_ec.contiguous(),
            w_schaffer.contiguous(),
            w_ta.contiguous(),
            y_ca1,
            mismatch,
            float(theta_rad),
            float(theta_inhibition),
            B=B,
            N_CA1=N_CA1,
            N_CA3=N_CA3,
            N_EC=N_EC,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
    else:
        # High-performance PyTorch CPU fallback
        g_ca3 = 0.5 * (1.0 - torch.cos(torch.tensor(theta_rad)))
        g_ec  = 0.5 * (1.0 + torch.cos(torch.tensor(theta_rad)))

        drive_ca3 = torch.matmul(x_ca3, w_schaffer.t())
        drive_ec  = torch.matmul(x_ec, w_ta.t())
        net = g_ca3 * drive_ca3 + g_ec * drive_ec - theta_inhibition
        y_ca1 = torch.relu(net)

    # Compute cosine mismatch score between normalized CA3 and EC projected drives
    drive_ca3_norm = torch.matmul(x_ca3, w_schaffer.t())
    drive_ec_norm  = torch.matmul(x_ec, w_ta.t())
    cos_sim = torch.cosine_similarity(drive_ca3_norm, drive_ec_norm, dim=-1)
    mismatch_score = 1.0 - cos_sim

    return y_ca1, mismatch_score
