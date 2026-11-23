import pytest
from fastapi.testclient import TestClient
from weft.api_server import app

client = TestClient(app)

def test_models_endpoint():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data["data"]) >= 5

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["active_models"] >= 10

def test_chat_completions_json():
    payload = {
        "model": "weft-auto",
        "messages": [{"role": "user", "content": "def fib(n):"}],
        "stream": False
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert "weft-coder" in data["choices"][0]["message"]["content"]
