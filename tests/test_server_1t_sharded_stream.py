import pytest
import subprocess
import time
import requests
import os
import signal

def test_server_1t_sharded_stream_endpoint():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8114
    cmd = ["./build/weft-server", model_path, str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.0)
    try:
        url = f"http://127.0.0.1:{port}/v1/models/1t/sharded_stream"
        resp = requests.get(url, timeout=3.0)
        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
        data = resp.json()
        assert data.get("model") == "weft-frontier-1.04T-moe"
        assert data.get("total_shards") == 16
        assert data.get("status") == "streaming_ready"
        print("[+] /v1/models/1t/sharded_stream verified successfully.")
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2.0)
