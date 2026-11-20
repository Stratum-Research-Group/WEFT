import pytest
import torch
import tiktoken
from weft.grammar_logits import JSONGrammarLogitsProcessor


def test_json_grammar_logits_processor_masking():
    tokenizer = tiktoken.get_encoding("gpt2")
    processor = JSONGrammarLogitsProcessor(tokenizer, vocab_size=50257)

    batch_size = 2
    vocab_size = 50257
    scores = torch.randn(batch_size, vocab_size)

    # When not inside tool call, scores should be unchanged
    unchanged = processor(torch.zeros(batch_size, 5, dtype=torch.long), scores.clone(), in_tool_call=False)
    assert torch.equal(scores, unchanged)

    # When inside tool call, invalid tokens must be masked to -inf
    constrained = processor(torch.zeros(batch_size, 5, dtype=torch.long), scores.clone(), in_tool_call=True)

    # Check that forbidden tokens are -inf
    invalid_indices = torch.where(~processor.valid_json_token_mask)[0]
    if len(invalid_indices) > 0:
        sample_idx = invalid_indices[0].item()
        assert constrained[0, sample_idx] == float("-inf")
        assert constrained[1, sample_idx] == float("-inf")

    # Check that valid structural tokens (e.g. '{', '"') are preserved
    lbrace_id = tokenizer.encode_ordinary("{")[0]
    assert constrained[0, lbrace_id] > float("-inf")
    assert constrained[0, lbrace_id] == scores[0, lbrace_id]
