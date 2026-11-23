import pytest
import numpy as np
from weft.mla_kv_cache import MLALowRankKVCache, MLAKVConfig
from weft.speculative_cascade import SpeculativeVerificationEngine

def test_mla_low_rank_kv_cache():
    cache = MLALowRankKVCache()
    k = np.random.randn(128, 4096).astype(np.float16)
    v = np.random.randn(128, 4096).astype(np.float16)
    
    stats = cache.append_and_compress(k, v)
    assert stats["compression_ratio"] >= 7.0  # 8x compression ratio
    
    k_rec, v_rec = cache.decompress_at_step(0)
    assert k_rec.shape == (128, 4096)
    assert v_rec.shape == (128, 4096)
    
    footprint = cache.total_memory_footprint()
    assert footprint["memory_reduction_factor"] == 8.0

def test_speculative_verification_cascade():
    engine = SpeculativeVerificationEngine(gamma=4)
    res = engine.speculative_decode_step("Test cascade prompt")
    
    assert res["accepted_tokens"] >= 1
    assert res["speedup_factor"] > 1.0  # Demonstrates net speedup over baseline
    
    bench = engine.benchmark_speculative_cascade(num_steps=5)
    assert bench["net_speedup"] > 1.2
