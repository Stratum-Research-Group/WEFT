"""
Test client for WEFT OpenAI-compatible server.
"""
import pytest
import torch
from fastapi.testclient import TestClient
from weft.serving_server import app, load_model

@pytest.fixture(scope="module")
def client():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model("weft_shakespeare_ckpt", device=device)
    return TestClient(app)

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"

def test_models_endpoint(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()["data"]
    assert len(data) > 0
    assert data[0]["id"] == "weft-causal-lm"

def test_chat_completions(client):
    req_body = {
        "model": "weft-causal-lm",
        "messages": [{"role": "user", "content": "KING:"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }
    res = client.post("/v1/chat/completions", json=req_body)
    assert res.status_code == 200
    json_data = res.json()
    assert "choices" in json_data
    assert len(json_data["choices"][0]["message"]["content"]) > 0

def test_chat_completions_streaming(client):
    req_body = {
        "model": "weft-causal-lm",
        "messages": [{"role": "user", "content": "ROMEO:"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
    }
    res = client.post("/v1/chat/completions", json=req_body)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    lines = [line for line in res.iter_lines() if line]
    assert any("[DONE]" in line for line in lines)
