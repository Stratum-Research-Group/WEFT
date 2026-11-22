import os
from pathlib import Path
import pytest
import torch
from weft.vlm_runner import SubBitVLMRunner

def test_vlm_runner_end_to_end():
    models_root = Path(os.environ.get("WEFT_MODELS_ROOT", "/run/media/akarta/AI/models"))
    weft_path = models_root / "lfm3-vl-2b-subbit" / "model_subbit.weft"
    if not weft_path.exists():
        pytest.skip(f"Shard file {weft_path} does not exist.")
    runner = SubBitVLMRunner(str(weft_path))
    assert runner.proj1 is not None
    assert runner.proj2 is not None

    # Input: 2 samples, 64 visual tokens (e.g. 8x8 patch grid), 1152 vision dim
    vision_feats = torch.randn(2, 64, runner.proj1.in_features)
    proj_out = runner.forward_vision_projection(vision_feats)
    
    assert proj_out.shape == (2, 64, runner.proj2.out_features)
    assert not torch.isnan(proj_out).any()

    # Multimodal prompt fusion test
    text_ids = torch.randint(0, 1000, (2, 16))
    fused = runner.forward_multimodal_fusion(vision_feats, text_ids)
    assert fused.shape == (2, 64 + 16, runner.proj2.out_features)
