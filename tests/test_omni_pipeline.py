import pytest
from weft.omni_pipeline import OmniMultimodalEdgePipeline, OmniRequest, OmniResponse

def test_omni_pipeline_initialization():
    pipeline = OmniMultimodalEdgePipeline()
    assert len(pipeline.verified_models) == 6
    for name, info in pipeline.verified_models.items():
        assert info["available"] is True, f"Model {name} GGUF not found!"
        assert info["size_mb"] > 0.0

def test_omni_pipeline_full_execution():
    pipeline = OmniMultimodalEdgePipeline()
    req = OmniRequest(
        audio_path="/tmp/audio.wav",
        image_path="/tmp/image.png",
        task_mode="full_omni"
    )
    res = pipeline.process(req)
    assert res.speech_transcription is not None
    assert res.visual_description is not None
    assert res.formal_proof_tree is not None
    assert res.code_solution is not None
    assert res.reasoning_trace is not None
    assert res.music_output_path is not None
    assert "total_omni_pipeline_latency_ms" in res.metrics

def test_omni_pipeline_math_mode():
    pipeline = OmniMultimodalEdgePipeline()
    req = OmniRequest(text_prompt="Calculate invariant", task_mode="math_reasoning")
    res = pipeline.process(req)
    assert res.formal_proof_tree is not None
    assert res.speech_transcription is None
