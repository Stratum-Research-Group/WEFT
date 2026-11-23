"""
Unit tests for Paper 64: 13-Model Co-Execution and Deep Adapter Retention Invariants in Lean 4.
"""

import subprocess
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from weft.api_server import app


def test_lean_deep_retention_invariants_compiles():
    """Verify formal/real_conjectures/DeepRetentionInvariants.lean compiles cleanly with 0 axioms and 0 sorry."""
    lean_file = Path("formal/real_conjectures/DeepRetentionInvariants.lean")
    assert lean_file.exists(), f"File {lean_file} missing"
    content = lean_file.read_text()
    assert "sorry" not in content
    assert "axiom" not in content

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean compilation error:\n{res.stderr}"


def test_api_server_13_model_fleet():
    """Verify the API server health and models endpoints report the full 13-model fleet."""
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["active_models"] == 13
    assert data["resident_ram_mb"] <= 2000.0
    assert data["retained_adapters_count"] >= 6


def test_chat_completions_llama_qwen_dispatch():
    """Verify chat completions dynamically route to 70B models."""
    client = TestClient(app)
    
    # Test Llama-70B routing
    res_llama = client.post("/v1/chat/completions", json={
        "model": "weft-auto",
        "messages": [{"role": "user", "content": "What is the general theory of frontier reasoning with llama?"}]
    })
    assert res_llama.status_code == 200
    assert "weft-llama70b" in res_llama.json()["choices"][0]["message"]["content"]

    # Test Qwen-72B routing
    res_qwen = client.post("/v1/chat/completions", json={
        "model": "weft-auto",
        "messages": [{"role": "user", "content": "Explain polyglot cross-lingual synthesis in qwen"}]
    })
    assert res_qwen.status_code == 200
    assert "weft-qwen72b" in res_qwen.json()["choices"][0]["message"]["content"]
