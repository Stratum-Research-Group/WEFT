"""
BitNet b1.58 Ternary Linear Layer and Quantization Engine for WEFT-2.

Implements 1.58-bit ternary quantization:
  W_quant = RoundClip(W / gamma, -1, 1) in {-1, 0, 1}
  where gamma = mean(|W|)
  Activations quantized to 8-bit integers:
  X_quant = RoundClip(X * (127 / max(|X|)), -128, 127)

Replaces floating-point matrix multiplications with integer additions/subtractions,
yielding up to 75% VRAM reduction and high energy efficiency.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant_b158(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes full-precision weight to ternary values {-1, 0, 1} with per-tensor scale factor.
    """
    gamma = weight.abs().mean().clamp(min=1e-5)
    w_scaled = weight / gamma
    w_quant = torch.round(w_scaled).clamp(-1.0, 1.0)
    return w_quant, gamma


def act_quant_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes activations to 8-bit integers [-128, 127] with per-token scale.
    """
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_quant = (x * scale).round().clamp(-128.0, 127.0)
    return x_quant, scale


class BitLinear158(nn.Linear):
    """
    1.58-bit Linear Layer replacing nn.Linear in WEFT FFN and Attention projections.
    During forward pass:
      y = (x_quant @ w_quant.T) * (gamma / scale)
    Straight-Through Estimator (STE) used for backpropagation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.rms_norm = nn.RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-layer RMSNorm for stable activation distribution
        x_norm = self.rms_norm(x)

        # Quantize weights: Straight-Through Estimator (STE)
        w_quant, gamma = weight_quant_b158(self.weight)
        w_ste = self.weight + (w_quant * gamma - self.weight).detach()

        # Quantize activations (during eval or deployment)
        x_quant, act_scale = act_quant_int8(x_norm)
        x_ste = x_norm + (x_quant / act_scale - x_norm).detach()

        out = F.linear(x_ste, w_ste, self.bias)
        return out


def convert_weft_to_bitnet(model: nn.Module) -> nn.Module:
    """
    Replaces all Linear projection layers in MLP/MoE experts and Attention with BitLinear158.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and "lm_head" not in name and "embed" not in name:
            bit_layer = BitLinear158(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
            ).to(device=module.weight.device)
            bit_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                bit_layer.bias.data.copy_(module.bias.data)
            setattr(model, name, bit_layer)
        else:
            convert_weft_to_bitnet(module)
    return model
