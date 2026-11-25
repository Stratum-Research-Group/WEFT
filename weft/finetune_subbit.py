"""
WEFT Block-Hessian Fine-Tuning Loop.
====================================
Optimizes continuous LittleBit-2 scale parameters (h, l, g) while freezing bipolar bitboards (U_b, V_b):
min_{h, l, g} || (W - sum_p diag(h_p) @ (U_b,p @ diag(l_p) @ V_b,p^T) @ diag(g_p)) X ||_F^2
Recovers perplexity and aligns sub-bit projections with activation distributions.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from weft.format import WeftSubBitLinear


def finetune_subbit_linear(
    layer: WeftSubBitLinear,
    target_W: torch.Tensor,
    sample_activations: torch.Tensor,
    num_steps: int = 25,
    lr: float = 1e-4,
) -> float:
    """
    Fine-tunes the continuous scale parameters of a WeftSubBitLinear layer
    against sample activations to minimize reconstruction error.
    sample_activations: [batch_size, in_features]
    target_W: [out_features, in_features]
    """
    device = target_W.device
    layer = layer.to(device)
    X = sample_activations.float().to(device)
    target_Y = torch.matmul(X, target_W.float().t())  # [B, out_features]

    # Convert scales to float32 for stable optimization
    orig_scales = []
    fp32_params = []
    for p in range(layer.num_paths):
        orig_scales.append((
            layer.scale_h[p].data.clone(),
            layer.scale_l[p].data.clone(),
            layer.scale_g[p].data.clone()
        ))
        # Keep fp32 trainable copies
        sh = nn.Parameter(layer.scale_h[p].data.float().clone())
        sl = nn.Parameter(layer.scale_l[p].data.float().clone())
        sg = nn.Parameter(layer.scale_g[p].data.float().clone())
        layer.scale_h[p] = sh
        layer.scale_l[p] = sl
        layer.scale_g[p] = sg
        fp32_params.extend([sh, sl, sg])

    optimizer = optim.Adam(fp32_params, lr=lr)
    criterion = nn.MSELoss()

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(num_steps):
        optimizer.zero_grad()
        pred_Y = layer(X)
        loss = criterion(pred_Y, target_Y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fp32_params, max_norm=1.0)
        optimizer.step()
        final_loss = loss.item()

    # Cast back to fp16
    for p in range(layer.num_paths):
        layer.scale_h[p] = nn.Parameter(layer.scale_h[p].data.to(torch.float16), requires_grad=False)
        layer.scale_l[p] = nn.Parameter(layer.scale_l[p].data.to(torch.float16), requires_grad=False)
        layer.scale_g[p] = nn.Parameter(layer.scale_g[p].data.to(torch.float16), requires_grad=False)

    return final_loss / max(initial_loss, 1e-6)
