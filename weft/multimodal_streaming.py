"""
WEFT-2 Native Streaming Multimodal (Vision & Audio) Adapter.

Maps continuous sensory feature embeddings (SigLIP vision patches, Whisper audio frames)
directly into WEFT's recurrent associative state space (S_t in R^{H_state x d_head x r}),
enabling continuous 24/7 video and audio stream ingestion with strictly O(1) constant VRAM.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from weft.configuration_weft import WeftConfig


class StreamingMultimodalProjector(nn.Module):
    """
    Projects external visual or acoustic tokens into WEFT's internal
    hidden representation and factorized state space.
    """

    def __init__(
        self,
        input_dim: int = 768,      # Standard SigLIP / Whisper latent dimension
        hidden_dim: int = 512,     # WEFT hidden_size
        state_dim: int = 64,       # WEFT head dimension
        state_rank: int = 16,      # WEFT associative rank
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Sensory state key/value projection for direct associative injection
        self.k_proj = nn.Linear(input_dim, state_dim)
        self.v_proj = nn.Linear(input_dim, state_rank)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        sensory_tokens: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Takes raw continuous sensory embeddings [B, S_sensory, D_in] and emits:
        1. Projected text-aligned tokens [B, S_sensory, hidden_dim]
        2. Direct associative keys k_s [B, S_sensory, state_dim]
        3. Direct associative values v_s [B, S_sensory, state_rank]
        """
        projected = self.norm(self.input_proj(sensory_tokens))
        k_s = self.k_proj(sensory_tokens)
        v_s = self.v_proj(sensory_tokens)
        return projected, k_s, v_s
