import pytest
import subprocess
import time
import requests
import os
import signal

def test_server_speculative_stream_sse_endpoint():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8118
    cmd = ["./build/weft-server", model_path, str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.0)
    try:
        url = f"http://127.0.0.1:{port}/v1/speculative/verify_tree_stream"
        payload = {"candidate_tokens": [100, 205, 304, 999]}
        resp = requests.post(url, json=payload, stream=True, timeout=3.0)
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                events.append(decoded)
        assert any("token_id" in ev for ev in events)
        assert any("[DONE]" in ev for ev in events)
        print("[+] /v1/speculative/verify_tree_stream SSE verified successfully.")
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2.0)
