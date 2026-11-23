import urllib.request
import json
import time
import subprocess
import os

def test_server_concurrent_requests():
    # Start temporary test server instance on port 8092
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    proc = subprocess.Popen(["./build/weft-server", model_path, "8092"])
    time.sleep(1.0)

    try:
        req = urllib.request.Request("http://127.0.0.1:8092/health")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"

        # Fire 5 parallel requests
        import concurrent.futures
        def fire_req(i):
            payload = {
                "model": "gemma-4-E2B-it-subbit",
                "messages": [{"role": "user", "content": f"Test concurrent token stream {i}"}],
                "stream": False
            }
            req = urllib.request.Request(
                "http://127.0.0.1:8092/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Weft-Adapter": "default"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fire_req, i) for i in range(5)]
            results = [f.result() for f in futures]

        assert len(results) == 5
        for res in results:
            assert "choices" in res
            assert len(res["choices"]) > 0
        print("[PASS] Concurrency load test passed across 5 simultaneous workers.")

    finally:
        proc.kill()
        proc.wait()

if __name__ == "__main__":
    test_server_concurrent_requests()
