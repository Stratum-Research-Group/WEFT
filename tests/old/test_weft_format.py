"""
Unit and integration tests for WEFT Binary Model Container Format (.weft).
Tests serialization, deserialization, bits-per-weight computation,
entropy coding, and numerical reconstruction equivalence.
"""

import os
import tempfile
import torch
import pytest
from weft.format import WeftSubBitLinear, WeftModelContainer


def test_weft_subbit_linear_forward():
    d_in, d_out, rank = 128, 256, 32
    layer = WeftSubBitLinear(in_features=d_in, out_features=d_out, rank=rank, num_paths=2, bias=True)
    
    x = torch.randn(2, 8, d_in)
    out = layer(x)
    
    assert out.shape == (2, 8, d_out)
    assert not torch.isnan(out).any()
    
    # Calculate bits per weight
    bpw = layer.bits_per_weight()
    assert 0.05 < bpw < 1.5, f"BPW {bpw} outside expected sub-bit regime"


def test_weft_container_save_load_roundtrip():
    container = WeftModelContainer(metadata={"architecture": "Weft-70B-SubBit", "hidden_size": 256, "vocab_size": 1000})
    
    layer1 = WeftSubBitLinear(in_features=128, out_features=256, rank=32, num_paths=2, bias=True)
    layer2 = WeftSubBitLinear(in_features=256, out_features=128, rank=16, num_paths=1, bias=False)
    
    container.add_subbit_linear("layers.0.mlp.up_proj", layer1)
    container.add_subbit_linear("layers.0.mlp.down_proj", layer2)
    
    with tempfile.NamedTemporaryFile(suffix=".weft", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # 1. Uncompressed save/load
        container.save(tmp_path, compress_entropy=False)
        loaded = WeftModelContainer.load(tmp_path)
        
        assert loaded.metadata["architecture"] == "Weft-70B-SubBit"
        assert "layers.0.mlp.up_proj" in loaded.tensors
        assert "layers.0.mlp.down_proj" in loaded.tensors
        
        reconstructed_layer1 = loaded.get_subbit_linear("layers.0.mlp.up_proj")
        
        x = torch.randn(2, 4, 128)
        y_orig = layer1(x)
        y_recon = reconstructed_layer1(x)
        
        assert torch.allclose(y_orig, y_recon, atol=1e-5), "Reconstructed layer output must match original exactly"
        
        # 2. Entropy compressed save/load
        container.save(tmp_path, compress_entropy=True)
        loaded_comp = WeftModelContainer.load(tmp_path)
        reconstructed_comp1 = loaded_comp.get_subbit_linear("layers.0.mlp.up_proj")
        y_comp = reconstructed_comp1(x)
        
        assert torch.allclose(y_orig, y_comp, atol=1e-5), "Entropy-compressed layer output must match original exactly"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
