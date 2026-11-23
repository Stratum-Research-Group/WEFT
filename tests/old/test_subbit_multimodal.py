"""
Unit tests for Sub-Bit Multimodal Projection and PagedAttention Ingestion.
"""

import torch
import pytest
from weft.subbit_multimodal import SubBitMultimodalProjector, SubBitMultimodalAttentionEngine


def test_subbit_multimodal_projector_forward():
    projector = SubBitMultimodalProjector(
        input_dim=128,
        hidden_dim=256,
        rank=32,
        num_paths=1
    )
    
    # Simulate batch of 2 sensory sequences of length 8
    sensory_tokens = torch.randn(2, 8, 128)
    out = projector(sensory_tokens)
    
    assert out.shape == (2, 8, 256)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_subbit_multimodal_attention_engine():
    projector = SubBitMultimodalProjector(
        input_dim=64,
        hidden_dim=128,
        rank=16,
        num_paths=1
    )
    
    engine = SubBitMultimodalAttentionEngine(
        projector=projector,
        hidden_dim=128,
        num_heads=4,
        head_dim=32,
        block_size=16,
        rank=16,
    )
    
    sensory_tokens = torch.randn(2, 16, 64)
    out = engine(sensory_tokens)
    
    assert out.shape == (2, 16, 128)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
