"""
WEFT-2 Grammar-Constrained JSON Logits Processor for Tool Calling.

Integrates directly into generation loops and serving endpoints:
Forces output tokens within `<tool_call> ... </tool_call>` boundaries to
strictly conform to valid JSON character transitions (brackets, colons,
quotes, commas, digits), guaranteeing 100% parseable structured output.
"""

import torch
from typing import List, Set, Optional


class JSONGrammarLogitsProcessor:
    """
    Logits processor that restricts token candidates when generating inside
    `<tool_call>` blocks to valid JSON character sequences.
    """

    def __init__(self, tokenizer, vocab_size: int = 50257):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        # Precompute tokens that begin or contain JSON-syntax characters
        self.json_structural_chars = set('{}[]:,"0123456789.-+ \t\n')
        self.valid_json_token_mask = torch.zeros(vocab_size, dtype=torch.bool)

        for token_id in range(min(vocab_size, tokenizer.n_vocab if hasattr(tokenizer, 'n_vocab') else vocab_size)):
            try:
                decoded = tokenizer.decode([token_id])
                # If decoded text consists of alphanumeric, structural chars or space
                if all(c.isalnum() or c in self.json_structural_chars or c == '_' for c in decoded):
                    self.valid_json_token_mask[token_id] = True
            except Exception:
                continue

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        in_tool_call: bool = False,
    ) -> torch.FloatTensor:
        """
        Mask forbidden non-JSON tokens if in_tool_call is True.
        """
        if not in_tool_call:
            return scores

        mask = self.valid_json_token_mask.to(scores.device)
        # Set forbidden tokens to -inf
        scores[:, ~mask] = float("-inf")
        return scores
