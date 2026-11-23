import pytest
from weft.speculative_coder import HierarchicalSubBitSpeculativeEngine, SpeculativeConfig

def test_speculative_coder_execution():
    engine = HierarchicalSubBitSpeculativeEngine()
    res = engine.generate("Implement fibonacci", max_new_tokens=30)
    assert res.total_tokens > 0
    assert res.speedup_factor > 1.0
    assert res.effective_tok_per_sec > 20.0
    assert res.metrics["resident_ram_mb"] < 250.0

def test_speculative_config_options():
    cfg = SpeculativeConfig(gamma_lookahead=2)
    engine = HierarchicalSubBitSpeculativeEngine(cfg)
    res = engine.generate("Test prompt", max_new_tokens=15)
    assert res.draft_tokens_proposed >= 0
