"""
Frontier Scaling Architecture Specifications for WEFT-2.

Defines exact configurations and verified parameter metrics across standard model tiers:
- 1.5B (Edge / Embedded / Speculative Draft)
- 3B   (Local Workstation / High-Efficiency Desktop)
- 7B   (Standard Open-Weights Benchmark Baseline)
- 14B  (Mid-Tier Heavy Reasoning Engine)
- 32B  (Frontier High-Capacity DeepSeek-style MoE)

Guarantees 0-byte host RAM allocation during instantiation using torch.device("meta").
"""

from typing import Dict, Tuple
from dataclasses import dataclass
import torch
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM


@dataclass(frozen=True)
class ModelSpec:
    tier: str
    target_params: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_state_heads: int
    state_rank: int
    window_size: int
    use_moe: bool
    moe_num_routed_experts: int
    moe_top_k: int
    use_mla: bool = True
    mla_latent_dim: int = 128
    tie_word_embeddings: bool = True

    def to_config(self) -> WeftConfig:
        return WeftConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_state_heads=self.num_state_heads,
            state_rank=self.state_rank,
            window_size=self.window_size,
            use_moe=self.use_moe,
            moe_num_routed_experts=self.moe_num_routed_experts,
            moe_top_k=self.moe_top_k,
            use_mla=self.use_mla,
            mla_latent_dim=self.mla_latent_dim,
            tie_word_embeddings=self.tie_word_embeddings,
        )


WEFT_SCALE_REGISTRY: Dict[str, ModelSpec] = {
    "1.5B": ModelSpec(
        tier="1.5B",
        target_params="1.48B",
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_state_heads=4,
        state_rank=32,
        window_size=128,
        use_moe=False,
        moe_num_routed_experts=0,
        moe_top_k=0,
    ),
    "3B": ModelSpec(
        tier="3B",
        target_params="3.01B",
        vocab_size=32000,
        hidden_size=2560,
        intermediate_size=7168,
        num_hidden_layers=42,
        num_attention_heads=20,
        num_state_heads=4,
        state_rank=32,
        window_size=128,
        use_moe=False,
        moe_num_routed_experts=0,
        moe_top_k=0,
    ),
    "7B": ModelSpec(
        tier="7B",
        target_params="6.99B",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_state_heads=4,
        state_rank=32,
        window_size=128,
        use_moe=False,
        moe_num_routed_experts=0,
        moe_top_k=0,
    ),
    "14B": ModelSpec(
        tier="14B",
        target_params="14.51B",
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=14336,
        num_hidden_layers=52,
        num_attention_heads=40,
        num_state_heads=8,
        state_rank=48,
        window_size=128,
        use_moe=False,
        moe_num_routed_experts=0,
        moe_top_k=0,
    ),
    "32B": ModelSpec(
        tier="32B",
        target_params="32.40B (MoE)",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=50,
        num_attention_heads=32,
        num_state_heads=8,
        state_rank=48,
        window_size=128,
        use_moe=True,
        moe_num_routed_experts=16,
        moe_top_k=2,
    ),
    "1T": ModelSpec(
        tier="1T",
        target_params="1.04T (MoE)",
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=88,
        num_attention_heads=64,
        num_state_heads=8,
        state_rank=64,
        window_size=128,
        use_moe=True,
        moe_num_routed_experts=128,
        moe_top_k=4,
    ),
}


def get_model_params(config: WeftConfig) -> Tuple[int, int]:
    """
    Computes total and active parameters strictly using torch.device('meta').
    Consumes zero bytes of host RAM or GPU VRAM.
    """
    with torch.device("meta"):
        model = WeftForCausalLM(config)
        total_params = sum(p.numel() for p in model.parameters())

    if not config.use_moe:
        active_params = total_params
    else:
        # Fine-grained DeepSeekMoE calculation:
        # shared expert is always active (d_shared)
        # routed experts: top_k are active out of num_routed_experts
        d_shared = max(64, int(config.intermediate_size * 0.5))
        d_shared = ((d_shared + 63) // 64) * 64
        shared_params = 3 * config.hidden_size * d_shared

        d_routed = max(64, int(config.intermediate_size * 0.5 / config.moe_top_k))
        d_routed = ((d_routed + 63) // 64) * 64
        single_routed_params = 3 * config.hidden_size * d_routed

        inactive_routed_count = config.moe_num_routed_experts - config.moe_top_k
        inactive_params = config.num_hidden_layers * inactive_routed_count * single_routed_params
        active_params = total_params - inactive_params

    return total_params, active_params
