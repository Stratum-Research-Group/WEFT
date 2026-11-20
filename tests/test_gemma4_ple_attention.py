"""
Unit test for Gemma 4 Per-Layer Embedding (PLE) and Attention Kernel Distillation.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.distill_gemma4_ple_and_attention import Gemma4AttentionPLEProber


GEMMA4_PATH = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"


@pytest.mark.skipif(not os.path.exists(GEMMA4_PATH), reason="Gemma 4 safetensors not found on external drive")
def test_gemma4_ple_and_attention_prober():
    prober = Gemma4AttentionPLEProber(GEMMA4_PATH)
    tokens = torch.tensor([[10, 20, 30], [40, 50, 60]])
    data = prober.probe_ple_and_attention(tokens, layer_idx=0)

    assert "ple_feature" in data
    assert "attn_out" in data
    assert data["ple_feature"].shape == (2, 3, 256), f"Unexpected PLE shape {data['ple_feature'].shape}"
    assert data["attn_out"].shape == (2, 3, 1536), f"Unexpected Attention shape {data['attn_out'].shape}"
    assert not torch.isnan(data["ple_feature"]).any()
    assert not torch.isnan(data["attn_out"]).any()
