"""Unit tests for Speculative Tree Decoder."""
import torch
import pytest
from weft import WeftConfig, WeftForCausalLM
from weft.speculative_tree_decoder import SpeculativeTreeDecoder

class DummyBridge(torch.nn.Module):
    def __init__(self, in_d=64, out_d=128):
        super().__init__()
        self.fc = torch.nn.Linear(in_d, out_d, bias=False)
    def forward(self, h, embed_w):
        return self.fc(h) @ embed_w.t()

def test_speculative_tree_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WeftConfig(vocab_size=32, hidden_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=32)
    model = WeftForCausalLM(cfg).to(device)
    bridge = DummyBridge(64, 128).to(device)
    embed_w = torch.randn(32, 128, device=device)

    decoder = SpeculativeTreeDecoder(
        draft_model=model,
        bridge=bridge,
        branch_factors=(2, 2), # 2 root branches, each branching into 2 -> 2 + 4 = 6 nodes
        device=device
    )

    prefix = torch.randint(0, 32, (1, 8), device=device)
    flat_tokens, tree_mask, nodes = decoder.generate_candidate_tree(prefix, embed_w)

    assert flat_tokens.shape == (1, 6)
    assert tree_mask.shape == (1, 1, 6, 6)
    assert len(nodes) == 6

    # Verify tree attention causality
    # Node 0 (depth 1) should only attend to itself
    assert tree_mask[0, 0, 0, 0] == 0.0
    assert tree_mask[0, 0, 0, 1] == float("-inf")

    # Node 2 (child of Node 0) should attend to itself and Node 0
    assert tree_mask[0, 0, 2, 2] == 0.0
    assert tree_mask[0, 0, 2, 0] == 0.0
    assert tree_mask[0, 0, 2, 1] == float("-inf")
