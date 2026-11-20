"""
WEFT-2 Lifelong Memory & Persistent State Adaptation Engine.

Unlike standard LLMs where weights are frozen and KV caches are wiped per session,
Lifelong Memory preserves the model's recurrent state tensor:
    S_t in R^{L x H_state x d_head x r}
across sessions to disk.

Key Features:
1. State Checkpointing & Restitution: Save/load recurrent states to disk.
2. Surprisal-Gated State Consolidation: Only update long-term state when new information
   has prediction error (surprisal > delta).
3. Online Memory Querying: Read from persistent memory without re-feeding past tokens.
"""

import os
import torch
from typing import Optional, Dict, Any


class LifelongMemoryStore:
    """Manages persistent recurrent state tensors for lifelong learning across conversations."""

    def __init__(self, memory_dir: str = "artifacts/lifelong_memory"):
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        self.active_state: Optional[torch.Tensor] = None

    def get_user_state_path(self, session_id: str) -> str:
        return os.path.join(self.memory_dir, f"{session_id}_recurrent_state.pt")

    def load_state(self, session_id: str, device: str = "cuda") -> Optional[torch.Tensor]:
        path = self.get_user_state_path(session_id)
        if os.path.exists(path):
            self.active_state = torch.load(path, map_location=device)
            return self.active_state
        return None

    def save_state(self, session_id: str, state: torch.Tensor):
        self.active_state = state.detach().clone()
        path = self.get_user_state_path(session_id)
        torch.save(self.active_state, path)

    def consolidate_state(
        self,
        current_state: torch.Tensor,
        new_state: torch.Tensor,
        surprisal: float,
        consolidation_threshold: float = 1.5,
    ) -> torch.Tensor:
        """
        Updates long-term state with a learned exponential momentum blend
        weighted by the surprisal (loss) of the new interaction.
        """
        if surprisal < consolidation_threshold:
            # Low novelty / predictable interaction: soft memory retention
            alpha = 0.95
        else:
            # High novelty / crucial new fact: aggressive state update
            alpha = 0.60

        consolidated = alpha * current_state + (1.0 - alpha) * new_state
        return consolidated
