"""Unit tests for MultiTaskSpecialistTreeRouter."""
import torch
import pytest
from weft import WeftConfig, WeftForCausalLM
from weft.mtp_head import MultiTokenPredictionHead
from weft.mtp_tree_integrator import MTPTreeIntegrator
from weft.tree_router_fusion import MultiTaskSpecialistTreeRouter

def test_tree_router_fusion_workflow():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WeftConfig(vocab_size=32, hidden_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=32)
    model = WeftForCausalLM(cfg).to(device)
    mtp = MultiTokenPredictionHead(student_dim=64, teacher_dim=128, num_future_tokens=4).to(device)
    tree_integrator = MTPTreeIntegrator(model, mtp, branch_factors=(2, 2), device=device)
    
    router = MultiTaskSpecialistTreeRouter(tree_integrator=tree_integrator, d_model=64, num_domains=7, device=device)
    embed_w = torch.randn(32, 128, device=device)
    prefix = torch.randint(0, 32, (1, 8), device=device)

    # Test specialized tree generation
    res = router.generate_specialized_tree(prefix, embed_w)
    assert "dispatched_domain" in res
    assert res["dispatched_domain"] in router.domains
    assert res["num_candidates"] == 6
    assert res["tree_mask"].shape == (1, 1, 6, 6)
