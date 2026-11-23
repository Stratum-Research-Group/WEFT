import pytest
import numpy as np
from weft.mla_kv_cache import MLALowRankKVCache, MLAKVConfig
from weft.speculative_cascade import SpeculativeVerificationEngine
from weft.adapter_router import HierarchicalAdapterRouter
from weft.multi_agent_council import MultiAgentCouncil
from weft.self_repair_engine import InRegisterSelfRepairEngine

def test_mla_low_rank_kv_cache():
    config = MLAKVConfig(d_model=512, n_heads=8, d_head=64, latent_dim=128, max_context=1024)
    cache = MLALowRankKVCache(config=config)
    k = np.random.randn(4, 512).astype(np.float32)
    v = np.random.randn(4, 512).astype(np.float32)
    res = cache.append_and_compress(k, v)
    assert res["tokens"] == 4
    assert res["compression_ratio"] > 1.0
    k_rec, v_rec = cache.decompress_at_step(0)
    assert k_rec.shape == (4, 512)
    assert v_rec.shape == (4, 512)
    stats = cache.total_memory_footprint()
    assert stats["memory_reduction_factor"] > 1.0

def test_speculative_verification_cascade():
    cascade = SpeculativeVerificationEngine()
    bench = cascade.benchmark_speculative_cascade(num_steps=10)
    assert bench["net_speedup"] > 1.0
    assert bench["total_tokens_generated"] > 0

def test_adapter_router_hierarchical():
    router = HierarchicalAdapterRouter()
    matches = router.route_query_to_adapter("quantum topology and manifold curvature", top_k=3)
    assert len(matches) > 0
    assert "adapter_id" in matches[0]
    assert "similarity_score" in matches[0]

def test_multi_agent_council():
    council = MultiAgentCouncil()
    res = council.deliberate_consensus("Verify convergence of spectral radius under rank-1 SVD projections.")
    assert "consensus" in res
    assert "total_latency_ms" in res
    assert len(res["stages"]) == 3

def test_self_repair_engine():
    engine = InRegisterSelfRepairEngine()
    res = engine.inject_and_repair(d_in=256, d_out=256)
    assert res["post_repair_fidelity"] >= 0.90
    assert res["recovery_success"] is True
