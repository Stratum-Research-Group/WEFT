"""
Adaptive Bit-Planes Allocation Engine (0.5 to 1.5 BPW).
========================================================
Dynamically assigns 1, 2, 3, or 4 bit-plane paths per linear projection
based on empirical singular value spectral decay gamma.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from weft.converter import estimate_spectral_decay, factorize_linear_layer
from weft.format import WeftSubBitLinear


def allocate_layer_bitplanes(gamma: float) -> int:
    """
    Decides the number of dual LittleBit-2 paths:
    - gamma < 0.4: slow singular decay (needs 4 paths = ~0.88 BPW)
    - 0.4 <= gamma < 0.7: moderate decay (needs 2 paths = ~0.44 BPW)
    - gamma >= 0.7: fast singular decay (needs 1 path = ~0.22 BPW)
    """
    if gamma < 0.4:
        return 4
    elif gamma < 0.7:
        return 2
    else:
        return 1


def factorize_with_adaptive_bitplanes(
    W: torch.Tensor,
    base_rank: int = 64,
) -> Tuple[WeftSubBitLinear, int, float]:
    """
    Factorizes matrix W with adaptive bitplane count.
    Returns: (subbit_linear, num_paths, gamma)
    """
    gamma = estimate_spectral_decay(W)
    num_paths = allocate_layer_bitplanes(gamma)
    layer = factorize_linear_layer(W, base_rank=base_rank, num_paths=num_paths)
    return layer, num_paths, gamma
