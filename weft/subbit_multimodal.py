"""
Sub-Bit Multimodal Projection Engine.
Integrates continuous sensory feature embeddings (SigLIP vision tokens, Whisper audio frames)
with the LittleBit-2 bipolar sub-bit projection heads and Sub-Bit PagedAttention engine.
Bypasses memory bandwidth limits by executing visual/audio token cross-projections directly
using packed bitboard linear operators.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from weft.format import WeftSubBitLinear
from weft.kernels.subbit_paged_attention import SubBitPagedAttention


class SubBitMultimodalProjector(nn.Module):
    """
    Sub-Bit Factorized Multimodal Projector.
    Projects sensory representations (Vision patches, Audio frames) into transformer hidden space
    using LittleBit-2 bipolar packed bitboards (0.23-0.54 BPW) with zero full-precision HBM staging.
    """
    def __init__(
        self,
        input_dim: int = 768,      # SigLIP patch / Whisper frame feature dimension
        hidden_dim: int = 2048,    # Foundation model d_model (e.g. Gemma-4 E2B hidden_size)
        rank: int = 64,            # Factorization rank r
        num_paths: int = 1,        # Dual-SVID path count
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_paths = num_paths

        # High-throughput sub-bit projection layers
        self.proj_in = WeftSubBitLinear(input_dim, hidden_dim, rank=rank, num_paths=num_paths, bias=False)
        self.act = nn.GELU()
        self.proj_out = WeftSubBitLinear(hidden_dim, hidden_dim, rank=rank, num_paths=num_paths, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sensory_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting continuous sensory tokens [B, S_sensory, input_dim]
        into transformer token space [B, S_sensory, hidden_dim].
        """
        h = self.proj_in(sensory_tokens)
        h = self.act(h)
        h = self.proj_out(h)
        return self.norm(h)


class SubBitMultimodalAttentionEngine(nn.Module):
    """
    Orchestrates continuous multimodal KV stream ingestion into SubBitPagedAttention.
    Stores audio/video tokens into sub-bit pages with O(1) constant VRAM overhead.
    """
    def __init__(
        self,
        projector: SubBitMultimodalProjector,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        block_size: int = 16,
        rank: int = 32,
    ):
        super().__init__()
        self.projector = projector
        self.attention = SubBitPagedAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            rank=rank,
        )

    def forward(self, sensory_tokens: torch.Tensor) -> torch.Tensor:
        """
        Projects sensory tokens and performs sub-bit self-attention directly.
        Returns:
            context_output: [B, S_sensory, hidden_dim]
        """
        projected = self.projector(sensory_tokens)
        out = self.attention(projected)
        return out
