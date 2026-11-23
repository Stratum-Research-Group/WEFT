import pytest
import os
from weft.audio_pipeline import MultimodalAudioPipeline

def test_multimodal_audio_pipeline():
    pipeline = MultimodalAudioPipeline()
    report = pipeline.execute_e2e_pipeline()
    
    assert report["whisper"]["status"] == "PROCESSED_NATIVE"
    assert "piano" in report["whisper"]["transcription"]
    assert report["thought"]["acoustic_conditioning"]["tempo_bpm"] > 0
    assert report["music"]["status"] == "SYNTHESIZED_OK"
    assert report["music"]["sample_rate"] == 32000
    assert report["e2e_latency_ms"] < 100.0  # Real-time interactive latency
