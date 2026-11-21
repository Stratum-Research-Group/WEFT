import torch
import pytest
from weft.serve_any import UniversalSubBitModel, load_universal_subbit_model

def test_universal_subbit_model_qwen():
    shard = "/run/media/akarta/AI/models/qwen-2.5-72b-subbit/model-00001-of-00037_subbit.weft"
    model = load_universal_subbit_model([shard])
    summary = model.summary()
    assert summary["total_layers"] == 7
    assert summary["effective_bpw"] < 0.1
    assert summary["compression_ratio"] > 100.0

    # Test forward pass on embedding/linear layer
    layer_name = "model.layers.0.mlp.gate_proj.weight"
    x = torch.randn(2, 8192)
    out = model.forward_layer(layer_name, x)
    assert out.shape == (2, 29568)
    assert not torch.isnan(out).any()

def test_universal_subbit_model_llama():
    shard = "/run/media/akarta/AI/models/llama-3.3-70b-subbit/model-00001-of-00030_subbit.weft"
    model = load_universal_subbit_model([shard])
    summary = model.summary()
    assert summary["total_layers"] == 13
    assert summary["effective_bpw"] < 0.1
    assert summary["compression_ratio"] > 100.0

    layer_name = "model.layers.0.mlp.gate_proj.weight"
    x = torch.randn(2, 8192)
    out = model.forward_layer(layer_name, x)
    assert out.shape == (2, 28672)
    assert not torch.isnan(out).any()
