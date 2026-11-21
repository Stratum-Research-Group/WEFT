"""Unit tests for MTPTreeIntegrator."""
import torch
import pytest
from weft import WeftConfig, WeftForCausalLM
from weft.mtp_head import MultiTokenPredictionHead
from weft.mtp_tree_integrator import MTPTreeIntegrator

def test_mtp_tree_single_shot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WeftConfig(vocab_size=32, hidden_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=32)
    model = WeftForCausalLM(cfg).to(device)
    mtp = MultiTokenPredictionHead(student_dim=64, teacher_dim=128, num_future_tokens=4).to(device)
    embed_w = torch.randn(32, 128, device=device)

    integrator = MTPTreeIntegrator(model, mtp, branch_factors=(2, 2), device=device)
    prefix = torch.randint(0, 32, (1, 8), device=device)

    flat_tokens, tree_mask, nodes = integrator.generate_candidate_tree_single_shot(prefix, embed_w)

    assert flat_tokens.shape == (1, 6)
    assert tree_mask.shape == (1, 1, 6, 6)
    assert len(nodes) == 6
    # Verify causality: node 1 is child of node 0 (so node 1 attends to node 0)
    assert tree_mask[0, 0, 1, 0] == 0.0
    # Node 0 is root (cannot attend to its child node 1)
    assert tree_mask[0, 0, 0, 1] == float("-inf")
