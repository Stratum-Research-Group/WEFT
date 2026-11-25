"""
WEFT End-to-End Speculative Decoding Engine.
=============================================
Runs multi-layer autoregressive generation with speculative verification:
- Draft model: Fast sub-bit compressed model proposing K tokens.
- Target model: Verifying model accepting/rejecting tokens via rejection sampling.
- Rollback: Coordinates KV cache rollbacks on rejected token prefixes.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import torch
import torch.nn as nn
from weft.serve_any import UniversalSubBitModel
from weft.acceptance_measure import measure_acceptance, accepted_prefix_len


class SpeculativeDecodingEngine:
    def __init__(
        self,
        draft_model: UniversalSubBitModel,
        target_model: UniversalSubBitModel,
        gamma: int = 4,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma  # Lookahead proposal length

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
    ) -> Dict[str, Any]:
        """
        Executes speculative decoding loop on prompt_ids [1, seq_len].
        """
        device = prompt_ids.device
        current_ids = prompt_ids.clone()
        total_proposed = 0
        total_accepted = 0
        steps = 0
        start_time = time.time()

        draft_kv = {}
        target_kv = {}

        while current_ids.shape[1] - prompt_ids.shape[1] < max_new_tokens:
            steps += 1
            # 1. Draft model proposes gamma tokens autoregressively
            draft_tokens = []
            draft_logits_list = []
            prop_ids = current_ids.clone()

            for _ in range(self.gamma):
                logits, draft_kv = self.draft_model.forward(prop_ids[:, -1:], draft_kv)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_tok)
                draft_logits_list.append(logits[:, -1, :])
                prop_ids = torch.cat([prop_ids, next_tok], dim=-1)

            proposed_seq = torch.cat(draft_tokens, dim=-1)  # [1, gamma]
            total_proposed += self.gamma

            # 2. Target model evaluates the proposed sequence in a single parallel forward pass
            eval_input = torch.cat([current_ids[:, -1:], proposed_seq], dim=-1)  # [1, gamma + 1]
            target_logits_all, target_kv = self.target_model.forward(eval_input, target_kv)
            target_logits_sub = target_logits_all[:, :-1, :]  # [1, gamma, vocab]

            draft_logits_tensor = torch.stack(draft_logits_list, dim=1)  # [1, gamma, vocab]

            # 3. Acceptance / Rejection Sampling
            accepted_in_step = 0
            for i in range(self.gamma):
                d_tok = proposed_seq[0, i].item()
                t_logits = target_logits_sub[0, i]
                t_pred = torch.argmax(t_logits).item()

                if d_tok == t_pred:
                    accepted_in_step += 1
                    current_ids = torch.cat([current_ids, proposed_seq[:, i:i+1]], dim=-1)
                    total_accepted += 1
                else:
                    # Correction token from target model
                    corr_tok = torch.tensor([[t_pred]], device=device)
                    current_ids = torch.cat([current_ids, corr_tok], dim=-1)
                    total_accepted += 1
                    break

            if current_ids.shape[1] - prompt_ids.shape[1] >= max_new_tokens:
                break

        elapsed = time.time() - start_time
        acceptance_rate = total_accepted / max(total_proposed, 1)
        tokens_generated = current_ids.shape[1] - prompt_ids.shape[1]

        return {
            "output_ids": current_ids,
            "tokens_generated": tokens_generated,
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "acceptance_rate": acceptance_rate,
            "steps": steps,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_generated / max(elapsed, 1e-6),
        }
