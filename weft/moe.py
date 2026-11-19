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

from typing import Optional, Tuple, Union
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

        # 3. Router Network with Associative Memory Routing option
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)
        self.use_associative_routing = True
        self.routing_key_proj = nn.Linear(d_model, 64, bias=False)
        self.routing_query_proj = nn.Linear(d_model, 64, bias=False)
        self.routing_decay = nn.Parameter(torch.tensor(2.5))  # sigmoid(2.5) ~ 0.923
        self.last_aux_loss = None

    def forward(
        self,
        x: torch.Tensor,
        router_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        x: [B, T, D] or [N, D]
        router_state: Optional associative routing memory matrix [B, 64, num_routed_experts]
        Returns:
          out: [B, T, D]
          next_router_state: updated associative routing matrix
        """
        orig_shape = x.shape
        is_3d = len(orig_shape) == 3
        if is_3d:
            b, s, d = orig_shape
            x_flat = x.view(-1, d)
        else:
            x_flat = x
            b, s, d = 1, x_flat.size(0), x_flat.size(-1)

        num_tokens = x_flat.size(0)

        # 1. Shared expert
        out_shared = self.shared_expert(x_flat)

        # 2. Router scores: static projection + associative memory readout
        router_logits_static = self.router(x_flat)  # [N, num_routed_experts]

        next_router_state = router_state
        if self.use_associative_routing and is_3d:
            # Associative needle-in-a-haystack routing
            q_r = F.normalize(self.routing_query_proj(x), dim=-1) # [B, S, 64]
            k_r = F.normalize(self.routing_key_proj(x), dim=-1)   # [B, S, 64]
            gamma_r = torch.sigmoid(self.routing_decay)

            if router_state is None:
                r_state = torch.zeros(b, 64, self.num_routed_experts, device=x.device, dtype=x.dtype)
            else:
                r_state = router_state

            # Step-by-step or parallel associative recall
            if s == 1:
                # O(1) decode fast path: read from current state, then update
                assoc_logits = torch.matmul(q_r, r_state).view(-1, self.num_routed_experts)
                router_logits = router_logits_static + 0.5 * assoc_logits

                # Accumulate memory from static assignment to match prefill update dynamics
                p_static = F.softmax(router_logits_static.view(b, 1, self.num_routed_experts), dim=-1)
                update = torch.bmm(k_r.transpose(1, 2), p_static)
                next_router_state = gamma_r * r_state + update
            else:
                # Sequence prefill with associative prefix scan across sequence
                # We compute associative historical bias for each token
                assoc_logits_list = []
                curr_state = r_state
                for t in range(s):
                    q_t = q_r[:, t : t + 1, :]
                    a_log = torch.matmul(q_t, curr_state).squeeze(1) # [B, num_experts]
                    assoc_logits_list.append(a_log)
                    k_t = k_r[:, t : t + 1, :]
                    # Use static router prediction for update to break sequential latency
                    p_static = F.softmax(router_logits_static.view(b, s, -1)[:, t : t + 1, :], dim=-1)
                    update_t = torch.bmm(k_t.transpose(1, 2), p_static)
                    curr_state = gamma_r * curr_state + update_t
                next_router_state = curr_state
                assoc_logits = torch.stack(assoc_logits_list, dim=1).view(-1, self.num_routed_experts)
                router_logits = router_logits_static + 0.5 * assoc_logits
        else:
            router_logits = router_logits_static

        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Select Top-K
        topk_weights, topk_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weights = topk_weights.to(x.dtype)

        # 3. Vectorized execution across routed experts
        out_routed = torch.zeros_like(x_flat)
        is_capturing = torch.cuda.is_current_stream_capturing() if torch.cuda.is_available() else False

        if is_capturing or (not self.training and x_flat.shape[0] <= 16):
            # Static graph-safe execution without CPU synchronization (mask.any() or dynamic where)
            for exp_idx, expert in enumerate(self.routed_experts):
                w = torch.where(topk_indices == exp_idx, topk_weights, torch.zeros_like(topk_weights)).sum(dim=-1, keepdim=True)
                expert_out = expert(x_flat)
                out_routed = out_routed + w * expert_out
        else:
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

        if is_3d and return_state:
            return out_flat.view(orig_shape), next_router_state
        return out_flat.view(orig_shape)

    def get_aux_loss(self) -> torch.Tensor:
        if self.last_aux_loss is not None:
            return self.last_aux_loss
        return torch.tensor(0.0, device=self.router.weight.device)
