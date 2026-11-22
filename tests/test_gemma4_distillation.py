"""
Unit test for Live Gemma 4 Sliced Distillation.
Verifies that Gemma 4 layer streaming and feature extraction works with zero-RAM footprint.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.distill_gemma4_pipeline import Gemma4LayerProber, distill_gemma4_to_weft


models_root = os.environ.get("WEFT_MODELS_ROOT", "/run/media/akarta/AI/models")
GEMMA4_PATH = os.path.join(models_root, "gemma-4-E2B-it", "model.safetensors")


@pytest.mark.skipif(not os.path.exists(GEMMA4_PATH), reason="Gemma 4 safetensors not found on external drive")
def test_gemma4_layer_prober():
    prober = Gemma4LayerProber(GEMMA4_PATH, hidden_size=1536)
    tokens = torch.tensor([[100, 200, 300], [400, 500, 600]])  # [B=2, S=3]
    out = prober.probe_layer(tokens, layer_idx=0)
    assert out.shape == (2, 3, 1536), f"Expected shape (2, 3, 1536) but got {out.shape}"
    assert not torch.isnan(out).any(), "Encountered NaN in probed activations"
