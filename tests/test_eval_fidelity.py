import pytest
from pathlib import Path
from weft.eval_fidelity import evaluate_model_health, evaluate_layer_reconstruction
from weft.serve_any import UniversalSubBitModel


def test_eval_fidelity_on_llama_shards():
    model_dir = Path("/run/media/akarta/AI/models/llama-3.3-70b-subbit")
    if not model_dir.exists():
        pytest.skip(f"Model directory {model_dir} does not exist.")

    shard_files = list(model_dir.glob("*.weft"))
    if not shard_files:
        pytest.skip("No .weft files in model directory.")

    report = evaluate_model_health(str(model_dir))
    assert report["all_layers_numerically_stable"] is True
    assert report["model_summary"]["total_linear_parameters"] > 40_000_000_000
    assert report["model_summary"]["effective_bpw"] < 0.04
    assert len(report["layer_evaluations"]) >= 1


def test_eval_layer_stability_synthetic(tmp_path):
    import torch
    from weft.format import WeftModelContainer
    from weft.converter import factorize_linear_layer

    # Create synthetic weight matrix and factorize
    W = torch.randn(256, 256, dtype=torch.float32)
    layer = factorize_linear_layer(W, base_rank=16)

    container = WeftModelContainer(metadata={"name": "test_synthetic"})
    container.add_subbit_linear("test.weight", layer)

    shard_path = tmp_path / "shard_00001.weft"
    container.save(str(shard_path))

    model = UniversalSubBitModel([str(shard_path)])
    res = evaluate_layer_reconstruction(model, "test.weight", batch_size=2, seq_len=16)
    assert res["is_numerically_stable"] is True
    assert res["energy_gain"] > 0.0
