"""
Low-Memory Custom Optimizers for WEFT.
Implements:
1. FusedBlockwise8BitAdamW: 8-bit quantized first and second momentum with dynamic block scaling (6 bytes/param vs 16 bytes/param).
2. MuonOptimizer: Momentum SGD with matrix-valued Newton-Schulz orthogonalization (Keller Jordan / Moonlight style, ~2 bytes/param).
Reduces optimizer VRAM footprint by 2.5x to 8x, enabling training 1B+ parameter models on single 16GB GPUs.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to orthogonalize matrix gradients G (polar decomposition).
    Used in Muon optimizer to accelerate convergence with low memory overhead.
    """
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    # Coefficients for order-5 Newton-Schulz polynomial
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz.
    Designed for 2D weight matrices (linear layers, projections).
    Maintains only 1st momentum (no 2nd variance matrix), reducing optimizer state to 2 bytes/param.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                # Apply Newton-Schulz orthogonalization for 2D matrices
                if update.ndim == 2 and update.size(0) >= 8 and update.size(1) >= 8:
                    ortho_update = newton_schulz5(update)
                    # Scale update by RMS
                    scale = max(1.0, update.size(0) / update.size(1)) ** 0.5
                    p.data.mul_(1.0 - lr * weight_decay)
                    p.data.add_(ortho_update, alpha=-lr * scale)
                else:
                    # Fallback for 1D biases or non-matrix tensors
                    p.data.mul_(1.0 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)

        return loss


class Quantized8BitAdamW(Optimizer):
    """
    Block-wise quantized 8-bit AdamW optimizer.
    Quantizes momentum (m) and variance (v) into int8 blocks of 64 or 128 elements,
    storing per-block float16 scale factors.
    Cuts optimizer VRAM from 16 bytes/param to ~6 bytes/param (2.6x savings).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, block_size=64):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            bs = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    # Store momentum and variance in INT8 with FP16 block scales
                    state["m_int8"] = torch.zeros(p.numel(), dtype=torch.int8, device=p.device)
                    state["m_scale"] = torch.zeros(math.ceil(p.numel() / bs), dtype=torch.float16, device=p.device)
                    state["v_int8"] = torch.zeros(p.numel(), dtype=torch.int8, device=p.device)
                    state["v_scale"] = torch.zeros(math.ceil(p.numel() / bs), dtype=torch.float16, device=p.device)

                state["step"] += 1
                step = state["step"]

                # Unpack and dequantize
                p_flat = p.view(-1)
                g_flat = grad.view(-1).float()

                # Dequantize m and v
                m_scale = state["m_scale"].repeat_interleave(bs)[:p.numel()]
                v_scale = state["v_scale"].repeat_interleave(bs)[:p.numel()]
                m = (state["m_int8"].float() / 127.0) * m_scale.float()
                v = (state["v_int8"].float() / 127.0) * v_scale.float()

                # Adam updates
                m.mul_(beta1).add_(g_flat, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_flat, g_flat, value=1.0 - beta2)

                # Re-quantize to int8
                m_reshaped = m.view(-1, bs) if p.numel() % bs == 0 else torch.nn.functional.pad(m, (0, bs - p.numel() % bs)).view(-1, bs)
                v_reshaped = v.view(-1, bs) if p.numel() % bs == 0 else torch.nn.functional.pad(v, (0, bs - p.numel() % bs)).view(-1, bs)

                new_m_scale = m_reshaped.abs().max(dim=1).values.clamp(min=1e-8)
                new_v_scale = v_reshaped.abs().max(dim=1).values.clamp(min=1e-8)

                state["m_scale"].copy_(new_m_scale.half())
                state["v_scale"].copy_(new_v_scale.half())

                norm_m = (m_reshaped / new_m_scale.unsqueeze(1)).clamp(-1.0, 1.0) * 127.0
                norm_v = (v_reshaped / new_v_scale.unsqueeze(1)).clamp(0.0, 1.0) * 127.0

                state["m_int8"].copy_(norm_m.view(-1)[:p.numel()].to(torch.int8))
                state["v_int8"].copy_(norm_v.view(-1)[:p.numel()].to(torch.int8))

                # Compute step update with bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                denom = v_hat.sqrt().add_(eps)
                step_size = lr

                if wd > 0:
                    p_flat.mul_(1.0 - lr * wd)

                p_flat.addcdiv_(m_hat, denom, value=-step_size)

        return loss
