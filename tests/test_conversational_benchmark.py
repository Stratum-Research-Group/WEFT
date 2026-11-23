import pytest
import json
import os

def test_conversational_quality_benchmark_results():
    report_path = "benchmarks/results/conversational_quality_benchmark.json"
    assert os.path.exists(report_path), "Conversational benchmark report must exist."
    
    with open(report_path, "r") as f:
        data = json.load(f)
        
    assert len(data["results"]) == 10
    assert data["metrics"]["total_tokens"] > 500
    assert data["metrics"]["avg_latency_ms"] < 100.0  # Ultra-fast edge dispatch
    for item in data["results"]:
        assert item["latency_ms"] > 0
        assert item["tokens"] > 0
        assert "weft-llama70b" in item["engine"]
