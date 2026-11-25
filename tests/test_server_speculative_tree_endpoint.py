import urllib.request
import json
import time
import subprocess
import os
import pytest

def test_server_speculative_tree_endpoint():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8109
    proc = subprocess.Popen(["./build/weft-server", model_path, str(port)])
    time.sleep(1.0)

    try:
        payload = json.dumps({
            "draft_tokens": [100, 205, 304, 408, 512, 620, 730, 840],
            "target_tokens": [100, 205, 304, 408, 999, 620, 730, 840],
            "parents": [-1, 0, 1, 2, 3, 4, 5, 6]
        }).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/speculative/verify_tree",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "verified"
            assert data["accepted_count"] == 4
            assert data["diverged_index"] == 4
            assert data["accepted_tokens"] == [100, 205, 304, 408]

        print("[PASS] /v1/speculative/verify_tree endpoint certified.")

    finally:
        proc.kill()
        proc.wait()

if __name__ == "__main__":
    test_server_speculative_tree_endpoint()
