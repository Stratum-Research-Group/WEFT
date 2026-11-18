"""
DeepSeekMoE Feed-Forward Network for WEFT.
Adapted from AETHER-1 (Specification v1.1.0).

Combines an always-active shared SwiGLU expert with fine-grained routed experts:
  y = FFN_shared(x) + sum_{i in TopK} g_i(x) * FFN_routed_i(x)

Benefits:
- Shared expert captures invariant syntax and global grammar.
- Fine-grained routed experts specialize in domain-specific tasks without parameter interference.
- Constant active parameter count per token (O(1) active FLOPs) while scaling total model capacity.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer."""
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class WeftDeepSeekMoE(nn.Module):
    """
    Fine-Grained Routed Experts with Always-Active Shared Expert for WEFT.
    """
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        num_routed_experts: int = 8,
        top_k: int = 2,
        shared_expert_ratio: float = 0.5,
        routed_expert_ratio: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_routed_experts = num_routed_experts
        self.top_k = min(top_k, num_routed_experts)

        # 1. Always-active Shared Expert
        d_shared = max(64, int(intermediate_size * shared_expert_ratio))
        d_shared = ((d_shared + 63) // 64) * 64
        self.shared_expert = SwiGLU(d_model, hidden_features=d_shared)

        # 2. Fine-grained Routed Experts
        d_routed = max(64, int(intermediate_size * routed_expert_ratio / self.top_k))
        d_routed = ((d_routed + 63) // 64) * 64
        self.routed_experts = nn.ModuleList([
            SwiGLU(d_model, hidden_features=d_routed) for _ in range(num_routed_experts)
        ])

        # 3. Router Network
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)
        self.last_aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.size(0)

        # 1. Shared expert
        out_shared = self.shared_expert(x_flat)

        # 2. Router scores
        router_logits = self.router(x_flat)  # [N, num_routed_experts]
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Select Top-K
        topk_weights, topk_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weights = topk_weights.to(x.dtype)

        # 3. Vectorized execution across routed experts
        out_routed = torch.zeros_like(x_flat)
        for exp_idx, expert in enumerate(self.routed_experts):
            mask = (topk_indices == exp_idx)
            if not mask.any():
                continue
            token_indices, k_slot = torch.where(mask)
            weights = topk_weights[token_indices, k_slot].unsqueeze(-1)
            expert_in = x_flat[token_indices]
            expert_out = expert(expert_in)
            out_routed.index_add_(0, token_indices, weights * expert_out)

        out_flat = out_shared + out_routed

        # 4. Auxiliary Load-Balancing Loss
        if self.training:
            f = torch.bincount(topk_indices.view(-1), minlength=self.num_routed_experts).float()
            f = f / (num_tokens * self.top_k)
            P = routing_probs.mean(dim=0)
            self.last_aux_loss = self.num_routed_experts * torch.sum(f * P)
        else:
            self.last_aux_loss = None

        return out_flat.view(orig_shape)

    def get_aux_loss(self) -> torch.Tensor:
        if self.last_aux_loss is not None:
            return self.last_aux_loss
        return torch.tensor(0.0, device=self.router.weight.device)
