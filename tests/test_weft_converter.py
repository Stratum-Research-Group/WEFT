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


def test_factorization_reconstruction_least_squares_convergence():
    """Verify that least-squares scale fitting achieves bounded relW and monotonically decreases error across paths."""
    torch.manual_seed(1)
    m, n, rank = 256, 256, 32
    gamma = 0.8
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    k = torch.arange(1, min(m, n) + 1).float()
    S = k ** (-gamma)
    W = (U * S.unsqueeze(0)) @ V.t()

    def get_reconstructed(layer):
        W_rec = torch.zeros(layer.out_features, layer.in_features)
        for p in range(layer.num_paths):
            u = layer.unpack_bitboard(layer.u_packed[p], layer.rank).float()
            v = layer.unpack_bitboard(layer.v_packed[p], layer.rank).float()
            h = layer.scale_h[p].float().unsqueeze(1)
            l = layer.scale_l[p].float()
            g = layer.scale_g[p].float().unsqueeze(0)
            W_rec += (h * (u * l.unsqueeze(0) @ v.t())) * g
        return W_rec

    layer_1p = factorize_linear_layer(W, base_rank=rank, adaptive_rank=False, num_paths=1, itq_iterations=10)
    layer_2p = factorize_linear_layer(W, base_rank=rank, adaptive_rank=False, num_paths=2, itq_iterations=10)

    W_rec_1p = get_reconstructed(layer_1p)
    W_rec_2p = get_reconstructed(layer_2p)

    norm_W = torch.norm(W)
    rel_1p = (torch.norm(W - W_rec_1p) / norm_W).item()
    rel_2p = (torch.norm(W - W_rec_2p) / norm_W).item()

    # RelW must be strictly bounded below 1.0 (better than zero-output) and decrease across paths
    assert rel_1p < 0.85, f"1-path relW {rel_1p} unexpectedly high"
    assert rel_2p < 0.70, f"2-path relW {rel_2p} unexpectedly high"
    assert rel_2p < rel_1p, f"2-path relW ({rel_2p}) must improve over 1-path ({rel_1p})"

