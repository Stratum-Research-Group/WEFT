"""
FastAPI Server Integration Verification (Option 3):
Tests live HTTP endpoints on weft/serving_server.py:
1. /health
2. /v1/specialized/catalog
3. /v1/chat/completions
4. /v1/specialized/predict
"""
import sys
import os
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft import WeftConfig, WeftForCausalLM, WeftCharTokenizer
from weft.serving_server import app, engine

def setup_mock_model():
    vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weft_shakespeare_ckpt", "vocab.json"))
    engine.tokenizer = WeftCharTokenizer(vocab_file=vocab_path)
    cfg = WeftConfig(
        vocab_size=engine.tokenizer.vocab_size,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine.model = WeftForCausalLM(cfg).to(device)
    engine.model.eval()
    engine.device = device

def test_endpoints():
    setup_mock_model()
    client = TestClient(app)

    
    print("=" * 70)
    print("  VERIFYING FASTAPI LIVE SERVING ENDPOINTS")
    print("=" * 70)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    print("  • GET /health -> 200 OK:", res.json())

    # 2. Specialized catalog
    res = client.get("/v1/specialized/catalog")
    assert res.status_code == 200
    catalog = res.json().get("catalog", [])
    print(f"  • GET /v1/specialized/catalog -> 200 OK: {len(catalog)} specialists registered.")

    # 3. Chat completion
    payload = {
        "model": "weft-2-sula",
        "messages": [{"role": "user", "content": "Hello WEFT!"}],
        "max_tokens": 8,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    print("  • POST /v1/chat/completions -> 200 OK:", res.json()["choices"][0]["message"])

    # 4. Specialized head prediction
    spec_payload = {
        "domain": "functiongemma",
        "input_text": "call weather_tool(location='Paris')",
    }
    res = client.post("/v1/specialized/predict", json=spec_payload)
    assert res.status_code == 200
    print("  • POST /v1/specialized/predict -> 200 OK:", res.json())

    print("=" * 70)
    print("  ALL FASTAPI SERVING ENDPOINTS FULLY VERIFIED (200 OK)!")
    print("=" * 70)

if __name__ == "__main__":
    test_endpoints()
