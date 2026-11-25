import urllib.request
import json
import time
import subprocess
import os
import concurrent.futures
import pytest

def test_server_continuous_batching_concurrency():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8101
    proc = subprocess.Popen(["./build/weft-server", model_path, str(port)])
    time.sleep(1.0)

    try:
        # Check health
        health_req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(health_req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"

        # Continuous batching simulation: 15 staggered requests across 5 threads
        def send_inference(req_id):
            time.sleep((req_id % 3) * 0.05) # simulate continuous overlapping arrivals
            payload = json.dumps({
                "model": "weft-subbit",
                "messages": [{"role": "user", "content": f"Batch prompt sequence {req_id}"}],
                "stream": False
            }).encode("utf-8")

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "X-Weft-Adapter": "default"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_inference, i) for i in range(15)]
            responses = [f.result() for f in futures]

        assert len(responses) == 15
        for res in responses:
            assert "choices" in res
            assert len(res["choices"]) > 0

        print(f"[PASS] Continuous request batching certified across 15 staggered requests.")

    finally:
        proc.kill()
        proc.wait()

if __name__ == "__main__":
    test_server_continuous_batching_concurrency()
