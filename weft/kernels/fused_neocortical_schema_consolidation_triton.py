"""
Neocortical Schema Assimilation & Systems Consolidation Kernel in Triton 3.9.

Models:
1. Fast Hippocampal / Subicular Episode Injection:
   Replay vector: x_replay ~ N(0, 1) or binary patterns from SWR events.
2. Slow Interleaved Neocortical Plasticity:
   delta_W_cortex = eta_cortex * (y_cortex @ x_replay^T - decay * W_cortex)
   where eta_cortex << eta_hippocampus (e.g. 0.01 vs 1.0)
3. Dual-Memory Inference:
   y_total = alpha * y_hippocampus + (1 - alpha) * y_cortex
   where alpha(t) = exp(-t / tau_consolidation) embodies the Ribot temporal gradient.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _interleaved_consolidation_kernel(
    W_CORTEX_ptr,      # [M, K] - neocortical synaptic weight matrix
    X_REPLAY_ptr,      # [B, K] - SWR compressed replay vectors
    Y_CORTEX_ptr,      # [B, M] - neocortical hidden representation
    DELTA_W_ptr,       # [M, K] - weight delta accumulator
    eta_cortex: float, # slow learning rate
    decay: float,      # weight decay
    B: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    # Accumulate outer product delta_W = eta * sum_b (y_b * x_b^T)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for b in range(B):
        # Load y[b, offs_m]
        y_b = tl.load(Y_CORTEX_ptr + b * M + offs_m, mask=mask_m, other=0.0)
        # Load x[b, offs_k]
        x_b = tl.load(X_REPLAY_ptr + b * K + offs_k, mask=mask_k, other=0.0)

        # Outer product: [BLOCK_M, 1] * [1, BLOCK_K]
        acc += y_b[:, None] * x_b[None, :]

    # Load current W_cortex
    w_curr = tl.load(
        W_CORTEX_ptr + offs_m[:, None] * K + offs_k[None, :],
        mask=(mask_m[:, None] & mask_k[None, :]),
        other=0.0
    )

    # Compute updated delta with weight decay
    delta = eta_cortex * acc - (eta_cortex * decay) * w_curr

    # Store delta
    tl.store(
        DELTA_W_ptr + offs_m[:, None] * K + offs_k[None, :],
        delta,
        mask=(mask_m[:, None] & mask_k[None, :])
    )


def neocortical_consolidation_step(
    w_cortex: torch.Tensor,
    x_replay: torch.Tensor,
    y_hpc: torch.Tensor,
    alpha_hpc: float = 0.5,
    eta_cortex: float = 0.01,
    decay: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs one step of interleaved neocortical systems consolidation.

    Args:
        w_cortex:   [M, K] slow neocortical connection matrix
        x_replay:   [B, K] incoming SWR replay pattern
        y_hpc:      [B, M] fast hippocampal episodic representation
        alpha_hpc:  relative hippocampal weight in combined recall
        eta_cortex: neocortical learning rate
        decay:      synaptic weight normalization decay

    Returns:
        (w_cortex_updated, y_combined, interference_metric)
    """
    B, K = x_replay.shape
    M = w_cortex.shape[0]

    # Forward pass through cortex: y_cortex = relu(x_replay @ W^T)
    y_cortex = torch.relu(torch.matmul(x_replay, w_cortex.t()))

    if x_replay.is_cuda:
        delta_w = torch.empty_like(w_cortex)
        BLOCK_M = 64 if M >= 64 else 16
        BLOCK_K = 64 if K >= 64 else 16
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

        _interleaved_consolidation_kernel[grid](
            w_cortex.contiguous(),
            x_replay.contiguous(),
            y_cortex.contiguous(),
            delta_w,
            float(eta_cortex),
            float(decay),
            B=B,
            M=M,
            K=K,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
        w_cortex_new = w_cortex + delta_w
    else:
        # High-performance PyTorch CPU fallback
        # delta_W = eta * (y_cortex.T @ x_replay) - eta * decay * W
        delta_w = eta_cortex * torch.matmul(y_cortex.t(), x_replay) - (eta_cortex * decay) * w_cortex
        w_cortex_new = w_cortex + delta_w

    # Combined system retrieval
    y_combined = alpha_hpc * y_hpc + (1.0 - alpha_hpc) * y_cortex

    # Catastrophic interference metric: norm of delta_w relative to w_cortex
    interference = (torch.norm(delta_w) / (torch.norm(w_cortex) + 1e-6)).item()

    return w_cortex_new, y_combined, interference
