"""
WEFT-2 Asymmetric Speculative Draft Engine.

Couples a high-speed, lightweight WEFT-2 model (<100M / BitNet 2-bit) as an ultra-fast
draft engine in front of large target frontier models (e.g., Llama-70B, Qwen-72B).

Drafts K tokens ahead via constant-memory linear recurrence, allowing the large
target model to verify K tokens in parallel in a single forward pass.
"""

import torch
from typing import List, Tuple, Callable


class AsymmetricSpeculator:
    """Orchestrates draft-then-verify speculative execution across asymmetric model scales."""

    def __init__(self, draft_model, draft_tokenizer, gamma: int = 4):
        self.draft_model = draft_model
        self.tokenizer = draft_tokenizer
        self.gamma = gamma

    def draft_tokens(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, List[int]]:
        """Autoregressively generates gamma draft candidate tokens using WEFT fast-path."""
        self.draft_model.eval()
        curr_ids = input_ids.clone()
        drafted_tokens = []

        with torch.no_grad():
            for _ in range(self.gamma):
                out = self.draft_model(curr_ids)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                drafted_tokens.append(next_tok.item())
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)

        return curr_ids, drafted_tokens

    def verify_tokens(
        self,
        candidate_ids: torch.LongTensor,
        num_drafted: int,
        target_verify_fn: Callable[[torch.LongTensor], torch.LongTensor],
    ) -> Tuple[torch.LongTensor, int]:
        """
        Submits full candidate sequence to target verifier.
        Accepts tokens where draft matches target prediction.
        """
        # Execute target model verification in parallel
        target_predictions = target_verify_fn(candidate_ids)

        accepted_count = 0
        prompt_len = candidate_ids.shape[1] - num_drafted

        for i in range(num_drafted):
            draft_token = candidate_ids[0, prompt_len + i]
            target_token = target_predictions[0, prompt_len + i - 1]
            if draft_token == target_token:
                accepted_count += 1
            else:
                break

        # Sequence accepted up to prompt_len + accepted_count
        accepted_ids = candidate_ids[:, : prompt_len + accepted_count]
        return accepted_ids, accepted_count
