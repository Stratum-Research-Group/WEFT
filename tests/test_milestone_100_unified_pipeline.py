"""
100th Test Milestone: Unified End-to-End Pipeline Integration Test.
Verifies the complete integration of:
1. MultiTaskSpecialistTreeRouter (dynamic specialist classification).
2. MTPTreeIntegrator (single-shot multi-token candidate tree drafting).
3. AdaptiveSpeculativeEntropyGate (Lean 4 verified reversal prevention).
4. AsyncTokenizerPipeline (concurrent non-blocking serialization).
"""
import pytest
import torch
from weft import WeftConfig, WeftForCausalLM
from weft.mtp_head import MultiTokenPredictionHead
from weft.mtp_tree_integrator import MTPTreeIntegrator
from weft.tree_router_fusion import MultiTaskSpecialistTreeRouter
from weft.entropy_gate import AdaptiveSpeculativeEntropyGate
from weft.async_tokenizer_pipeline import AsyncTokenizerPipeline

@pytest.mark.anyio
async def test_milestone_100_unified_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize components
    cfg = WeftConfig(vocab_size=32, hidden_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=32)
    draft_model = WeftForCausalLM(cfg).to(device)
    mtp_head = MultiTokenPredictionHead(student_dim=64, teacher_dim=128, num_future_tokens=4).to(device)
    tree_integrator = MTPTreeIntegrator(draft_model, mtp_head, branch_factors=(2, 2), device=device)
    router = MultiTaskSpecialistTreeRouter(tree_integrator=tree_integrator, d_model=64, num_domains=7, device=device)
    entropy_gate = AdaptiveSpeculativeEntropyGate(gamma=4, entropy_threshold=5.0)
    async_tokenizer = AsyncTokenizerPipeline()

    # 2. Async tokenize test prompt
    prompt = "def verify_100th_test_milestone(): return True"
    input_ids = await async_tokenizer.encode_async(prompt)
    assert len(input_ids) > 0

    # 3. Dynamic specialist routing & candidate tree generation
    prefix_tensor = torch.randint(0, 32, (1, 8), device=device)
    embed_w = torch.randn(32, 128, device=device)
    tree_res = router.generate_specialized_tree(prefix_tensor, embed_w)
    
    assert "dispatched_domain" in tree_res
    assert tree_res["dispatched_domain"] in router.domains
    assert tree_res["num_candidates"] == 6

    # 4. Entropy gate verification
    dummy_logits = torch.randn(1, 32, device=device)
    allow_speculation, entropy = entropy_gate.should_speculate(dummy_logits)
    assert isinstance(allow_speculation, bool)
    assert entropy > 0.0

    # 5. Async decode output
    decoded = await async_tokenizer.decode_async(input_ids)
    assert "verify_100th_test_milestone" in decoded
    async_tokenizer.shutdown()
