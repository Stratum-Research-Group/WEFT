"""
Test suite for WEFT sub-bit model converter and Joint-ITQ factorization.
"""

import os
import tempfile
import torch
import pytest
from weft.converter import factorize_linear_layer, convert_model_to_weft, estimate_spectral_decay
from weft.format import WeftModelContainer


def test_spectral_decay_estimation():
    # Construct a matrix with power-law singular values
    U, _ = torch.linalg.qr(torch.randn(64, 64))
    V, _ = torch.linalg.qr(torch.randn(64, 64))
    k = torch.arange(1, 65).float()
    S = 10.0 * (k ** -0.8)
    W = U @ torch.diag(S) @ V.t()
    
    gamma = estimate_spectral_decay(W)
    assert 0.5 < gamma < 1.2, f"Estimated decay {gamma} outside expected range for true gamma=0.8"


def test_factorize_linear_layer():
    m, n = 128, 256
    W = torch.randn(m, n)
    bias = torch.randn(m)
    
    subbit = factorize_linear_layer(
        W, bias=bias, base_rank=32, adaptive_rank=False, num_paths=2, itq_iterations=5
    )
    
    assert subbit.in_features == n
    assert subbit.out_features == m
    assert subbit.rank == 32
    assert subbit.bias is not None
    
    x = torch.randn(2, 4, n)
    y_approx = subbit(x)
    assert y_approx.shape == (2, 4, m)
    assert not torch.isnan(y_approx).any()


def test_convert_model_to_weft_roundtrip():
    state_dict = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        "model.layers.0.mlp.down_proj.weight": torch.randn(64, 128),
        "model.layers.0.mlp.down_proj.bias": torch.randn(64),
    }
    config = {"hidden_size": 64, "vocab_size": 1000, "num_layers": 1}
    
    with tempfile.NamedTemporaryFile(suffix=".weft", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        convert_model_to_weft(state_dict, config, tmp_path, base_rank=32, adaptive_rank=True)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0
        
        loaded = WeftModelContainer.load(tmp_path)
        assert "model.layers.0.mlp.gate_proj" in loaded.tensors
        assert "model.layers.0.mlp.down_proj" in loaded.tensors
        
        gate_layer = loaded.get_subbit_linear("model.layers.0.mlp.gate_proj")
        assert gate_layer.in_features == 64
        assert gate_layer.out_features == 128
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
