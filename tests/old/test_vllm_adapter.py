"""
Unit test for WEFT-2 vLLM Worker Adapter.
"""

import pytest
import torch
from weft.configuration_weft import WeftConfig
from weft.vllm_adapter import WeftVLLMWorkerAdapter, register_with_vllm


def test_vllm_worker_adapter_batching():
    cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=16,
        window_size=32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = WeftVLLMWorkerAdapter(cfg, max_num_seqs=8, device=device)

    # Step batch across 3 sequences
    seq_ids = [101, 102, 103]
    tokens = [42, 88, 12]
    logits = adapter.step_batch(seq_ids, tokens)

    assert logits.shape == (3, 1000)
    assert not torch.isnan(logits).any()

    # Free sequences
    for sid in seq_ids:
        adapter.free_sequence(sid)
    assert len(adapter.active_sequences) == 0


def test_vllm_registration_hook():
    res = register_with_vllm()
    assert isinstance(res, bool)
