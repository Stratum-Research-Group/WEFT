"""
Scale Residual Distillation Engine:
Optimizes floating-point channel scales (h, g) and rank scales (l)
to minimize output reconstruction error against dense reference activations,
keeping binary bitboards (U_b, V_b) frozen. Zero additional storage overhead.
"""

import torch
import torch.nn as nn
from typing import Optional
from weft.format import WeftSubBitLinear


def tune_scale_residuals(
    subbit_layer: WeftSubBitLinear,
    W_dense: torch.Tensor,
    num_steps: int = 50,
    lr: float = 1e-2,
    batch_size: int = 16,
    seq_len: int = 32,
) -> float:
    """
    Optimizes scale_h, scale_l, scale_g via AdamW on synthetically generated
    or real calibration activations to minimize MSE: ||y_subbit - x @ W_dense^T||^2.
    Returns final relative L2 reconstruction error.
    """
    device = W_dense.device
    subbit_layer.to(device)

    # Freeze bitboards
    for p in range(subbit_layer.num_paths):
        subbit_layer.u_packed[p].requires_grad = False
        subbit_layer.v_packed[p].requires_grad = False
        subbit_layer.scale_h[p].requires_grad = True
        subbit_layer.scale_l[p].requires_grad = True
        subbit_layer.scale_g[p].requires_grad = True

    # Gather scale parameters
    scale_params = []
    for p in range(subbit_layer.num_paths):
        scale_params.extend([subbit_layer.scale_h[p], subbit_layer.scale_l[p], subbit_layer.scale_g[p]])

    optimizer = torch.optim.AdamW(scale_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    W_float = W_dense.float()

    for step in range(num_steps):
        # Calibration inputs
        x = torch.randn(batch_size, seq_len, subbit_layer.in_features, device=device)
        y_ref = torch.matmul(x, W_float.t())

        y_pred = subbit_layer(x).float()

        loss = torch.nn.functional.mse_loss(y_pred, y_ref)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Final evaluation
    with torch.no_grad():
        x_val = torch.randn(batch_size, seq_len, subbit_layer.in_features, device=device)
        y_val_ref = torch.matmul(x_val, W_float.t())
        y_val_pred = subbit_layer(x_val).float()
        
        rel_l2 = (torch.norm(y_val_pred - y_val_ref) / torch.norm(y_val_ref)).item()

    return rel_l2
