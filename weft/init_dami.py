"""
Dual-Axis Memory Initialization (DAMI) for WEFT Gated Recurrence.
Adapted from AETHER-1 / arXiv:2610.41994 (Feng et al., 2026).

Spaces decay logits geometrically across heads and rank channels so that
different state channels natively specialize across historical horizons:
from short local syntax (tau ~ 64 tokens) to ultra-long horizons (tau ~ 65,536 tokens).
"""

import math
import torch


def init_dami_decay_logits(
    decay_tensor: torch.Tensor,
    min_timescale: float = 64.0,
    max_timescale: float = 65536.0,
) -> None:
    """
    Initializes WEFT decay_logits parameter with geometrically spaced timescales.
    
    WEFT compute: gamma = sigmoid(decay_logits) in (0, 1).
    Effective retention timescale: tau = -1.0 / ln(gamma).
    For gamma close to 1: gamma approx 1 - 1/tau.
    Since gamma = sigmoid(z) = 1 / (1 + exp(-z)),
    z = ln(gamma / (1 - gamma)) approx ln(tau - 1).
    """
    with torch.no_grad():
        orig_shape = decay_tensor.shape
        num_elements = decay_tensor.numel()
        
        if num_elements == 1:
            mid_tau = math.sqrt(min_timescale * max_timescale)
            gamma = math.exp(-1.0 / mid_tau)
            decay_tensor.fill_(math.log(gamma / (1.0 - gamma)))
            return

        # Logarithmically spaced timescales from min_timescale to max_timescale
        log_min = math.log(min_timescale)
        log_max = math.log(max_timescale)
        step = (log_max - log_min) / max(1, num_elements - 1)
        timescales = torch.exp(torch.arange(num_elements, dtype=torch.float32) * step + log_min)
        
        # gamma = exp(-1.0 / tau)
        gammas = torch.exp(-1.0 / timescales)
        # Clamp to avoid numerical overflow in logit
        gammas = torch.clamp(gammas, min=0.50, max=0.9999)
        logits = torch.log(gammas / (1.0 - gammas))
        
        decay_tensor.copy_(logits.view(orig_shape))
