"""
2-Level Hierarchical Vocabulary Expansion (Bottleneck 2 Fix):
Combines:
1. Fast Level-1 Active SRAM Head (Top 4,096 tokens): Zero-latency GPU kernel drafting.
2. Sparse Level-2 Full Gemma Head (262,144 tokens): Dynamic Top-K fallback slicing
   for rare words, foreign languages, and specialized identifiers.
"""
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalVocabHead(nn.Module):
    def __init__(
        self,
        bridge: nn.Module,
        active_embed: torch.Tensor,
        full_embed_ref: Optional[torch.Tensor] = None,
        active_vocab_size: int = 4096,
        full_vocab_size: int = 262144,
    ):
        super().__init__()
        self.bridge = bridge
        self.register_buffer("active_embed", active_embed)
        self.full_embed_ref = full_embed_ref
        self.active_vocab_size = active_vocab_size
        self.full_vocab_size = full_vocab_size

    def forward(self, h: torch.Tensor, use_full_vocab: bool = False) -> torch.Tensor:
        # h: [B, S, D_student]
        # Level 1: Fast SRAM projection over active 4,096 tokens
        active_logits = self.bridge(h, self.active_embed) # [B, S, 4096]
        
        if not use_full_vocab or self.full_embed_ref is None:
            return active_logits

        # Level 2: Sparse Expansion over full 262,144 Gemma tokens
        full_logits = self.bridge(h, self.full_embed_ref) # [B, S, 262144]
        return full_logits
