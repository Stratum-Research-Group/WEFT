"""
Cerebellar Granule Layer Sparse Expansion & Golgi Cell Feedback Inhibition Kernel in Triton 3.9.

Models:
1. Massive Divergent Projections (N_MF -> N_GrC, ~100x ratio):
   Each granule cell possesses exactly 4 dendritic claws receiving excitatory inputs
   from 4 distinct mossy fiber rosettes.
2. Recurrent Golgi Cell Feedback Inhibition:
   Golgi cells sample parallel fiber excitation (W_grc_to_golgi) and deliver broad
   shunting GABAergic feedback onto mossy fiber glomeruli (I_golgi).
3. Population Sparsity Enforcement:
   Enforces < 1.0% population sparsity via competitive thresholding:
   y_GrC = relu( W_MF_4claws @ x_MF - I_golgi - theta_GrC )
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _cerebellar_granule_kernel(
    X_MF_ptr,          # [B, N_MF] - mossy fiber input vectors
    CLAWS_IDX_ptr,     # [N_GRC, 4] - 4 mossy fiber indices per granule cell
    CLAWS_WEIGHT_ptr,  # [N_GRC, 4] - synaptic weights for each claw
    I_GOLGI_ptr,       # [B] - global/regional Golgi cell feedback inhibition
    Y_GRC_ptr,         # [B, N_GRC] - output granule cell firing rate
    theta_grc: float,  # granule cell firing threshold
    B: tl.constexpr,
    N_GRC: tl.constexpr,
    N_MF: tl.constexpr,
    BLOCK_GRC: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)

    offs_g = pid_g * BLOCK_GRC + tl.arange(0, BLOCK_GRC)
    mask_g = offs_g < N_GRC

    # Load Golgi inhibition scalar
    i_golgi = tl.load(I_GOLGI_ptr + pid_b)

    # Accumulate 4 dendritic claw inputs for each granule cell in parallel
    acc = tl.zeros([BLOCK_GRC], dtype=tl.float32)

    for claw in range(4):
        # Load claw index for this granule cell
        idx = tl.load(CLAWS_IDX_ptr + offs_g * 4 + claw, mask=mask_g, other=0)
        w = tl.load(CLAWS_WEIGHT_ptr + offs_g * 4 + claw, mask=mask_g, other=0.0)

        # Load mossy fiber value
        mf_val = tl.load(X_MF_ptr + pid_b * N_MF + idx, mask=mask_g, other=0.0)
        acc += w * mf_val

    # Subtract Golgi shunting inhibition and threshold
    net_drive = acc - i_golgi - theta_grc
    y_grc = tl.maximum(net_drive, 0.0)

    # Store granule cell activation
    tl.store(Y_GRC_ptr + pid_b * N_GRC + offs_g, y_grc, mask=mask_g)


def cerebellar_granule_expansion(
    x_mf: torch.Tensor,
    claws_idx: torch.Tensor,
    claws_weight: torch.Tensor,
    i_golgi: torch.Tensor,
    theta_grc: float = 0.8
) -> Tuple[torch.Tensor, float]:
    """
    Simulates divergent mossy fiber 4-claw expansion into the granular layer under Golgi feedback.

    Args:
        x_mf:         [B, N_MF] mossy fiber inputs
        claws_idx:    [N_GRC, 4] integer indices of 4 connecting mossy fibers
        claws_weight: [N_GRC, 4] synaptic weights of the 4 claws
        i_golgi:      [B] Golgi cell shunting inhibition per batch element
        theta_grc:    activation threshold

    Returns:
        (y_grc, active_fraction)
    """
    B, N_MF = x_mf.shape
    N_GRC = claws_idx.shape[0]
    device = x_mf.device

    if x_mf.is_cuda:
        y_grc = torch.empty((B, N_GRC), device=device, dtype=torch.float32)
        BLOCK_GRC = 128 if N_GRC >= 128 else 32
        grid = (B, triton.cdiv(N_GRC, BLOCK_GRC))

        _cerebellar_granule_kernel[grid](
            x_mf.contiguous(),
            claws_idx.contiguous(),
            claws_weight.contiguous(),
            i_golgi.contiguous(),
            y_grc,
            float(theta_grc),
            B=B,
            N_GRC=N_GRC,
            N_MF=N_MF,
            BLOCK_GRC=BLOCK_GRC,
        )
    else:
        # High-performance PyTorch CPU fallback
        # Gather 4 claw inputs: shape [B, N_GRC, 4]
        # x_mf: [B, N_MF]
        gathered = x_mf[:, claws_idx] # [B, N_GRC, 4]
        # Multiply weights and sum over claws
        acc = torch.sum(gathered * claws_weight.unsqueeze(0), dim=-1) # [B, N_GRC]
        net_drive = acc - i_golgi.unsqueeze(-1) - theta_grc
        y_grc = torch.relu(net_drive)

    # Active population fraction
    active_mask = (y_grc > 0.0).float()
    active_fraction = float(active_mask.mean().item())

    return y_grc, active_fraction
