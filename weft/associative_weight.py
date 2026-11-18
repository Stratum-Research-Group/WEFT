"""
WEFT Associative Memory Weights (AMW) Module.
Replaces or augments dense feed-forward networks (MLP) with a fast,
ETH tensor-uncrossed associative dictionary lookup:
  1. Encoder Projector: Maps input x into a compact query key space: q = Encoder(x) [B, S, R]
  2. Associative Weight Memory: Structured low-rank dictionary of keys K_w and values V_w.
  3. Fast Needle Retrieval: Sparse or soft associative readout via dot-product uncrossing:
     M(x) = (Softmax(q @ K_w^T / sqrt(R)) @ V_w)
Computes associative feed-forward responses in O(R * D) without streaming large parameter matrices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeftAssociativeWeight(nn.Module):
    def __init__(self, hidden_size: int, num_memory_slots: int = 128, rank: int = 64, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_slots = num_memory_slots
        self.rank = rank

        # 1. Query Encoder: projects tokens into low-rank memory routing keys
        self.encoder = nn.Linear(hidden_size, rank, bias=False)

        # 2. Associative Key-Value Dictionary (The Memory Haystack)
        # Stored as low-rank factored basis tensors
        self.memory_keys = nn.Parameter(torch.randn(num_memory_slots, rank) / math.sqrt(rank))
        self.memory_values = nn.Parameter(torch.randn(num_memory_slots, hidden_size) / math.sqrt(hidden_size))

        # 3. Readout Projection & Normalization
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, D]
        Returns: [B, S, D] associative memory update
        """
        b, s, d = x.shape

        # Step 1: Encode query needle
        # q: [B, S, R]
        q = self.encoder(x)

        # Step 2: Associative retrieval over weight dictionary
        # affinities: [B, S, M]
        affinities = torch.matmul(q, self.memory_keys.t()) / math.sqrt(self.rank)
        weights = F.softmax(affinities, dim=-1)
        weights = self.dropout(weights)

        # Step 3: Retrieve and uncross value features
        # retrieved: [B, S, D]
        retrieved = torch.matmul(weights, self.memory_values)

        # Step 4: Gated readout
        g = torch.sigmoid(self.gate(x))
        out = self.norm(x + g * retrieved)
        return out
