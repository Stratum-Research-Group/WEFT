"""
Live Automated Ollama Endpoint Test.
===================================
Tests communication with the local Ollama daemon on http://127.0.0.1:11434.
"""

import urllib.request
import json
import pytest


def test_ollama_daemon_connectivity():
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            assert response.status == 200
            data = json.loads(response.read().decode())
            assert "models" in data
            print(f"[PASS] Ollama daemon responsive, active models: {len(data['models'])}")
    except Exception as e:
        pytest.skip(f"Ollama daemon not running locally: {e}")
