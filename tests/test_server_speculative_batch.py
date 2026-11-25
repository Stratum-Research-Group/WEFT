import pytest
import subprocess
import time
import requests
import os
import signal

def test_server_speculative_batch_endpoint():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8116
    cmd = ["./build/weft-server", model_path, str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.0)
    try:
        url = f"http://127.0.0.1:{port}/v1/speculative/verify_tree_batch"
        payload = {
            "streams": [
                {"stream_id": 0, "candidate_tokens": [100, 205, 304, 408]},
                {"stream_id": 1, "candidate_tokens": [99, 101, 102]}
            ]
        }
        resp = requests.post(url, json=payload, timeout=3.0)
        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
        data = resp.json()
        assert data.get("object") == "speculative_batch_verification"
        assert data.get("batch_size") == 2
        assert len(data.get("stream_results", [])) == 2
        assert data.get("status") == "batch_verified"
        print("[+] /v1/speculative/verify_tree_batch verified successfully.")
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2.0)
