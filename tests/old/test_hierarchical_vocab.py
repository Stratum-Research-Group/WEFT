"""Unit tests for 2-Level Hierarchical Vocabulary Head."""
import torch
import pytest
from weft.hierarchical_vocab import HierarchicalVocabHead

class MockBridge(torch.nn.Module):
    def forward(self, h, embed):
        return h @ embed.t()

def test_hierarchical_vocab_levels():
    bridge = MockBridge()
    active_embed = torch.randn(64, 128)
    full_embed = torch.randn(512, 128)

    vocab_head = HierarchicalVocabHead(
        bridge=bridge,
        active_embed=active_embed,
        full_embed_ref=full_embed,
        active_vocab_size=64,
        full_vocab_size=512,
    )

    h = torch.randn(2, 4, 128)

    # Level 1 fast active
    logits_l1 = vocab_head(h, use_full_vocab=False)
    assert logits_l1.shape == (2, 4, 64)

    # Level 2 full vocabulary
    logits_l2 = vocab_head(h, use_full_vocab=True)
    assert logits_l2.shape == (2, 4, 512)
