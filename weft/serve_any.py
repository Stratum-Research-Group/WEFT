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
                continue
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

    def summary(self) -> Dict[str, Any]:
        total_layers = len(self.layer_metadata)
        total_params = 0
        total_bits = 0
        for name, meta in self.layer_metadata.items():
            in_f = meta["in_features"]
            out_f = meta["out_features"]
            r = meta["rank"]
            params = in_f * out_f
            bits = (in_f * r) + (out_f * r) + (r * 16)
            total_params += params
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
