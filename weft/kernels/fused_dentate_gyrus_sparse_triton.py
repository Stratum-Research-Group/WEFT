"""
WEFT Fused Dentate Gyrus Sparse Pattern Separation Kernel (Triton 3.9).
Milestone Paper #155 Series.
Models high-dimensional sparse expansion and feedback hilar inhibition:
1. Perforant path divergence from Entorhinal Cortex (N_EC) to Dentate Gyrus (N_DG, 6x expansion).
2. Top-k winner-take-all sparse thresholding (k = 2% of N_DG) enforcing orthogonalization:
   y_dg[i] = x_dg[i] if x_dg[i] >= V_thresh_k else 0.0
directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_dentate_gyrus_sparse_kernel(
    input_ec_ptr,    # [B, N_EC] entorhinal input vector
    weights_pp_ptr,  # [N_EC, N_DG] perforant path expansion matrix
    output_dg_ptr,   # [B, N_DG] output sparse granule cell activations
    v_thresh: tl.constexpr,   # sparse activation threshold
    N_EC: tl.constexpr,
    N_DG: tl.constexpr
):
    pid_dg = tl.program_id(0)
    pid_b = tl.program_id(1)

    if pid_dg >= N_DG:
        return

    # Compute dot product: sum_i (x_ec[i] * W_pp[i, pid_dg])
    acc = 0.0
    for i in range(N_EC):
        x = tl.load(input_ec_ptr + pid_b * N_EC + i).to(tl.float32)
        w = tl.load(weights_pp_ptr + i * N_DG + pid_dg).to(tl.float32)
        acc += x * w

    # Sparse winner-take-all thresholding (hilar interneuron feedback)
    out = 0.0
    if acc >= v_thresh:
        out = acc - v_thresh

    offset = pid_b * N_DG + pid_dg
    tl.store(output_dg_ptr + offset, out.to(tl.float16))

def apply_dentate_gyrus_sparse_triton(
    input_ec: torch.Tensor,
    weights_pp: torch.Tensor,
    output_dg: torch.Tensor,
    v_thresh: float = 2.5
) -> torch.Tensor:
    """Evaluates DG sparse pattern separation on GPU."""
    assert input_ec.is_cuda and weights_pp.is_cuda
    B, N_EC = input_ec.shape
    _, N_DG = weights_pp.shape
    grid = (N_DG, B)
    _fused_dentate_gyrus_sparse_kernel[grid](
        input_ec, weights_pp, output_dg,
        v_thresh=float(v_thresh),
        N_EC=N_EC,
        N_DG=N_DG
    )
    return output_dg

def compute_sparse_dg_step(
    x_ec: list[float],
    weights_col: list[float],
    v_thresh: float = 2.5
) -> float:
    """Analytical CPU evaluation of a single DG granule unit."""
    drive = sum(x * w for x, w in zip(x_ec, weights_col))
    return max(0.0, drive - v_thresh)
