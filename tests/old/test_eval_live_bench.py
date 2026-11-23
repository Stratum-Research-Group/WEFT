import pytest
from pathlib import Path
from weft.fleet_status import inspect_fleet_status
from weft.eval_live_bench import benchmark_model_inference


def test_fleet_status_inspect():
    models_root = Path("/run/media/akarta/AI/models")
    if not models_root.exists():
        pytest.skip(f"Models root {models_root} does not exist.")

    report = inspect_fleet_status(str(models_root))
    assert "llama-3.3-70b-subbit" in report
    llama_data = report["llama-3.3-70b-subbit"]
    assert llama_data["completed_shards"] >= 20
    assert llama_data["compression_factor"] > 400.0


def test_eval_live_bench_execution():
    model_dir = Path("/run/media/akarta/AI/models/llama-3.3-70b-subbit")
    if not model_dir.exists():
        pytest.skip("Llama subbit model dir not found.")

    res = benchmark_model_inference(str(model_dir), max_new_tokens=4)
    assert res["all_layers_healthy"] is True
    assert res["tokens_per_second"] > 0.0
    assert res["shards_count"] >= 20
