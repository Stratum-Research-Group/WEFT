import pytest
from weft.eval_kv_subbit import benchmark_kv_cache


def test_eval_kv_subbit():
    res = benchmark_kv_cache(
        num_layers=8,
        max_blocks=64,
        block_size=16,
        latent_dim=192,
        simulated_steps=16,
    )
    assert res["status"] == "PASS"
    assert res["kv_compression_ratio"] > 4.0
    assert res["mean_token_kv_write_ms"] < 5.0
