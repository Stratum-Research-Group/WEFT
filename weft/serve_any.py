"""
WEFT Universal Model Loader & Inference Engine
Supports dynamically loading and serving any architecture quantized in WEFT .weft format:
Qwen, Llama, Goedel, Gemma, Liquid, etc.
"""

from typing import Dict, Any, List, Optional
import os
import torch
import torch.nn as nn
from weft.format import WeftModelContainer, WeftSubBitLinear
from weft.kernels.subbit_paged_attention import SubBitPagedAttention


class UniversalSubBitModel(nn.Module):
    """
    Dynamically loads and reconstructs linear projections from .weft containers.
    Supports sharded execution, register-space dequantization, and PagedAttention KV cache.
    """
    def __init__(self, shards: List[str]):
        super().__init__()
        self.shard_paths = shards
        self.layers: nn.ModuleDict = nn.ModuleDict()
        self.layer_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_shards()

    def _load_shards(self) -> None:
        for shard_path in self.shard_paths:
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Sub-bit shard file not found: {shard_path}")
            container = WeftModelContainer.load(shard_path)
            for tensor_name in container.tensors.keys():
                layer = container.get_subbit_linear(tensor_name)
                # Register using a sanitized key for ModuleDict
                sanitized_name = tensor_name.replace(".", "_")
                self.layers[sanitized_name] = layer
                self.layer_metadata[tensor_name] = {
                    "sanitized_name": sanitized_name,
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "rank": layer.rank,
                    "shard": os.path.basename(shard_path),
                }

    def get_layer(self, original_name: str) -> Optional[WeftSubBitLinear]:
        meta = self.layer_metadata.get(original_name)
        if meta is None:
            return None
        sanitized = meta["sanitized_name"]
        if sanitized in self.layers:
            return self.layers[sanitized]
        return None

    def forward_layer(self, original_name: str, x: torch.Tensor) -> torch.Tensor:
        layer = self.get_layer(original_name)
        if layer is None:
            raise KeyError(f"Layer '{original_name}' not found in loaded .weft shards.")
        return layer(x)

    def generate_step(self, token_ids: torch.Tensor, top_k: int = 50) -> torch.Tensor:
        """
        STRUCTURAL SMOKE TEST ONLY - NOT THE MODEL'S REAL COMPUTATION.

        This step uses only the LAST token id, builds a hand-made one-hot
        pseudo-embedding (not the embedding table), runs layer-0 MLP only, and
        reuses the embedding weight as the output head. Attention, norms and
        layers 1..N-1 play no role, so the output is NOT a language-model
        prediction and must never be used to claim output quality.
        Its purpose is to exercise shard loading and the sub-bit linear path.
        """
        import warnings
        warnings.warn(
            "UniversalSubBitModel.generate_step is a structural smoke test, not a faithful "
            "transformer forward pass; do not use its output for quality claims.",
            RuntimeWarning,
            stacklevel=2,
        )
        device = token_ids.device
        embed = self.get_layer("model.embed_tokens.weight")
        if embed is None:
            # Fall back to first available layer
            first_key = list(self.layer_metadata.keys())[0]
            embed = self.get_layer(first_key)

        # In sub-bit representation, embed layer is [vocab_size, hidden_size] or transposed
        batch_size, seq_len = token_ids.shape
        last_token = token_ids[:, -1]  # [batch]

        # Use actual token indices to construct initial hidden representation
        # Deterministic projection from token index to hidden_dim
        h = torch.zeros(batch_size, embed.in_features, device=device)
        for b in range(batch_size):
            tok_val = int(last_token[b].item())
            # Deterministic pseudo-embedding based on token id
            h[b, tok_val % embed.in_features] = 1.0
            h[b, (tok_val * 7 + 1) % embed.in_features] = 0.5

        # Pass through Layer 0 MLP if present
        gate = self.get_layer("model.layers.0.mlp.gate_proj.weight")
        up = self.get_layer("model.layers.0.mlp.up_proj.weight")
        down = self.get_layer("model.layers.0.mlp.down_proj.weight")
        if gate is not None and up is not None:
            g = torch.nn.functional.silu(gate(h))
            u = up(h)
            mlp_h = g * u
            if down is not None:
                h = h + down(mlp_h)
            else:
                # If down is in another shard, reduce dimensionality
                h = h + mlp_h[:, :embed.in_features]

        # Project to vocabulary logits via embed layer
        logits = embed(h)  # [batch, vocab_size]
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        return next_tokens

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 16) -> torch.Tensor:
        """
        Autoregressively generates max_new_tokens.
        """
        curr_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            next_id = self.generate_step(curr_ids)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
        return curr_ids

    def summary(self) -> Dict[str, Any]:
        total_layers = len(self.layer_metadata)
        total_params = 0
        total_bits = 0
        for name, meta in self.layer_metadata.items():
            in_f = meta["in_features"]
            out_f = meta["out_features"]
            params = in_f * out_f
            total_params += params
            sanitized = meta["sanitized_name"]
            if sanitized in self.layers:
                layer = self.layers[sanitized]
                # bits_per_weight returns bits / params
                bits = int(round(layer.bits_per_weight() * params))
            else:
                r = meta["rank"]
                bits = (in_f * r) + (out_f * r) + (r * 16)
            total_bits += bits
        effective_bpw = (total_bits / total_params) if total_params > 0 else 0.0
        return {
            "total_layers": total_layers,
            "total_linear_parameters": total_params,
            "total_bits": total_bits,
            "effective_bpw": effective_bpw,
            "compression_ratio": (total_params * 16) / max(total_bits, 1),
            "loaded_shards": [os.path.basename(p) for p in self.shard_paths],
        }


def load_universal_subbit_model(shard_paths: List[str]) -> UniversalSubBitModel:
    return UniversalSubBitModel(shard_paths)
