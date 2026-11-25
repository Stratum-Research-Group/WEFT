"""
Unit and regression test suite for Triton Rectified Flow DiT step,
C++ AVX2 Audio DSP resampler microbenchmark, and Multimodal & STEM fleet sync.
"""
import pytest
import subprocess
import torch
from pathlib import Path
from weft.kernels.fused_dit_triton import rectified_flow_step_triton
from weft.audio_pipeline import MultimodalAudioPipeline

def test_rectified_flow_step_triton_cuda():
    """Verify Triton Rectified Flow DiT step matches reference Euler step on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton DiT test.")
        
    device = torch.device("cuda:0")
    b, n, d = 2, 64, 128
    x = torch.randn((b, n, d), dtype=torch.float16, device=device)
    v = torch.randn((b, n, d), dtype=torch.float16, device=device)
    dt = 0.25

    out_triton = rectified_flow_step_triton(x, v, dt=dt)
    out_ref = (x.float() + dt * v.float()).half()

    max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
    assert max_diff < 1e-3, f"DiT step error exceeded tolerance: {max_diff}"

def test_audio_pipeline_stereo_44k():
    """Verify upgraded audio pipeline generates 44.1kHz stereo audio."""
    pipe = MultimodalAudioPipeline()
    res = pipe.synthesize_music_rvq({"key": "D minor"}, duration_s=1.0)
    assert res["sample_rate"] == 44100
    assert res["channels"] == 2
    assert res["total_frames"] == 44100
    assert res["status"] == "SYNTHESIZED_OK"

def test_weft_bench_audio_dsp_binary():
    """Verify build/weft-bench-audio-dsp binary executes and reports real-time speedup."""
    bin_path = Path("build/weft-bench-audio-dsp")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-audio-dsp binary not compiled.")
        
    cmd = [str(bin_path), "100"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Speedup Factor" in result.stdout
    assert "Realtime" in result.stdout
