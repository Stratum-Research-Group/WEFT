"""Unit tests for WeftVLLMTreeAdapter."""
import torch
import pytest
from weft import WeftConfig, WeftForCausalLM
from weft.speculative_tree_decoder import SpeculativeTreeDecoder
from weft.vllm_tree_adapter import WeftVLLMTreeAdapter

class MockBridge(torch.nn.Module):
    def forward(self, h, embed):
        return h @ embed.t()

def test_vllm_tree_adapter_workflow():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WeftConfig(vocab_size=32, hidden_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=32)
    model = WeftForCausalLM(cfg).to(device)
    bridge = MockBridge().to(device)
    embed_w = torch.randn(32, 64, device=device)

    tree_decoder = SpeculativeTreeDecoder(model, bridge, branch_factors=(2, 2), device=device)
    adapter = WeftVLLMTreeAdapter(tree_decoder=tree_decoder)

    prefix = torch.randint(0, 32, (1, 8), device=device)
    inputs = adapter.prepare_speculative_tree_inputs(prefix, embed_w)

    assert "draft_token_ids" in inputs
    assert "tree_attention_mask" in inputs
    assert inputs["num_candidates"] == 6

    # Verify export config
    cfg_meta = adapter.export_vllm_tree_config()
    assert cfg_meta["speculative_mode"] == "tree_attention"
    assert cfg_meta["expected_acceptance_alpha"] == 0.8775
