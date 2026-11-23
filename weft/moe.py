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
                # Vectorized chunked associative prefix scan across sequence (O(1) kernel launches)
                t_idx = torch.arange(s, device=x.device, dtype=torch.float32)
                p_static_all = F.softmax(router_logits_static.view(b, s, self.num_routed_experts), dim=-1)

                # 1. State initialization contribution: q_r[:, t] @ (gamma^t * r_state)
                pow_gamma = torch.pow(gamma_r, t_idx).view(1, s, 1)
                assoc_init = torch.matmul(q_r * pow_gamma.to(q_r.dtype), r_state)

                # 2. In-sequence pairwise historical contribution:
                # For long sequences, evaluate in chunks of 1024 tokens to bound transient VRAM
                chunk_size = 1024
                assoc_updates = []
                for c_start in range(0, s, chunk_size):
                    c_end = min(c_start + chunk_size, s)
                    q_c = q_r[:, c_start:c_end, :]
                    dots_c = torch.bmm(q_c, k_r[:, :c_end, :].transpose(1, 2))
                    c_pos = t_idx[c_start:c_end].unsqueeze(1)
                    k_pos = t_idx[:c_end].unsqueeze(0)
                    diff = c_pos - k_pos - 1
                    valid_mask = diff >= 0
                    safe_diff = diff.clamp(min=0.0)
                    decay_mat = torch.where(valid_mask, torch.pow(gamma_r, safe_diff), torch.zeros_like(diff))
                    M_c = dots_c * decay_mat.to(q_r.dtype)
                    assoc_c = torch.bmm(M_c, p_static_all[:, :c_end, :])
                    assoc_updates.append(assoc_c)

                assoc_update = torch.cat(assoc_updates, dim=1) if len(assoc_updates) > 1 else assoc_updates[0]
                assoc_logits = (assoc_init + assoc_update).view(-1, self.num_routed_experts)
                router_logits = router_logits_static + 0.5 * assoc_logits

                # 3. Final state transition for subsequent decode:
                # S_final = gamma^s * r_state + sum_{j=0}^{s-1} gamma^{s - 1 - j} (k_r[j]^T * p[j])
                gamma_s = torch.pow(gamma_r, s).view(1, 1, 1).to(q_r.dtype)
                decay_final = torch.pow(gamma_r, s - 1 - t_idx).view(1, s, 1).to(q_r.dtype)
                k_weighted = k_r * decay_final
                final_update = torch.bmm(k_weighted.transpose(1, 2), p_static_all)
                next_router_state = r_state * gamma_s + final_update
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


class PolytopeMoERouter(nn.Module):
    """
    Polytope Mixture-of-Experts Router with Convex Facet Boundary Stability.

    Partitions representation space into convex polyhedral cells defined by
    facet hyperplanes and centroid distance metrics:
      P_e = {x in R^d : A_e x <= b_e}

    Guarantees that tokens with facet margin > 2 * delta maintain strictly
    invariant expert assignments under sub-bit low-rank quantization.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        facet_margin_threshold: float = 0.05,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.facet_margin_threshold = facet_margin_threshold
        self.temperature = temperature

        # Learnable facet normal centroids (hyperplane normals)
        self.facet_centroids = nn.Parameter(torch.empty(num_experts, d_model))
        # Learnable per-facet bias / hyperplane thresholds
        self.facet_bias = nn.Parameter(torch.zeros(num_experts))
        nn.init.orthogonal_(self.facet_centroids)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x: Input token representations [..., d_model]

        Returns:
          topk_weights: Normalized routing weights [..., top_k]
          topk_indices: Selected expert indices [..., top_k]
          facet_margins: Margin between top-1 and runner-up facet projections [...]
        """
        orig_shape = x.shape[:-1]
        x_flat = x.view(-1, self.d_model)

        # Normalize token embeddings and facet centroids for robust angular distance
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        c_norm = F.normalize(self.facet_centroids, p=2, dim=-1)

        # Facet projections: A_e x + b_e
        projections = torch.matmul(x_norm, c_norm.T) + self.facet_bias  # [N, num_experts]

        # Compute facet margins (top-1 projection minus top-2 projection)
        sorted_proj, _ = torch.sort(projections, dim=-1, descending=True)
        margins = sorted_proj[:, 0] - sorted_proj[:, 1]

        # Apply geometric facet margin guard:
        # If margin < facet_margin_threshold, damp routing uncertainty by smoothing probability mass toward centroid
        scaled_logits = projections / self.temperature
        if self.facet_margin_threshold > 0.0:
            fragile_mask = (margins < self.facet_margin_threshold).unsqueeze(-1)
            # Soften logits for tokens located near polytope decision boundary
            scaled_logits = torch.where(fragile_mask, scaled_logits * 0.5, scaled_logits)

        routing_probs = F.softmax(scaled_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return (
            topk_weights.view(*orig_shape, self.top_k),
            topk_indices.view(*orig_shape, self.top_k),
            margins.view(orig_shape)
        )

