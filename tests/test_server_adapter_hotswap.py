import urllib.request
import json
import time
import subprocess
import os
import concurrent.futures
import pytest

def test_server_adapter_hotswap_stress():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8098
    proc = subprocess.Popen(["./build/weft-server", model_path, str(port)])
    time.sleep(1.0)

    try:
        # 1. Check /v1/models endpoint
        req_models = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
        with urllib.request.urlopen(req_models, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["object"] == "list"
            assert len(data["data"]) >= 1

        # 2. Dynamically load adapters
        for adapt in ["math_adapter", "coder_adapter", "prover_adapter"]:
            load_payload = json.dumps({"id": adapt}).encode("utf-8")
            load_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/adapters/load",
                data=load_payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(load_req, timeout=2.0) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                assert res["status"] == "loaded"

        # 3. Check loaded list
        req_adapters = urllib.request.Request(f"http://127.0.0.1:{port}/v1/adapters")
        with urllib.request.urlopen(req_adapters, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            adapter_ids = [item["id"] for item in data["data"]]
            assert "math_adapter" in adapter_ids
            assert "coder_adapter" in adapter_ids

        # 4. Multi-threaded concurrent requests with diverse adapter configurations
        def fire_request(thread_idx):
            headers = {"Content-Type": "application/json"}
            if thread_idx % 3 == 0:
                headers["X-Weft-Adapter"] = "math_adapter"
            elif thread_idx % 3 == 1:
                headers["X-Weft-Adapters"] = "math_adapter=0.6,coder_adapter=0.4"
            else:
                headers["X-Weft-Adapter"] = "default"

            payload = json.dumps({
                "model": "weft-subbit",
                "messages": [{"role": "user", "content": f"Query from thread {thread_idx}"}],
                "stream": False
            }).encode("utf-8")

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=payload,
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fire_request, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert len(results) == 20
        for r in results:
            assert "choices" in r

        # 5. Unload one adapter
        unload_payload = json.dumps({"id": "math_adapter"}).encode("utf-8")
        unload_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/adapters/unload",
            data=unload_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(unload_req, timeout=2.0) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            assert res["status"] == "unloaded"

        print("[PASS] Dynamic adapter hotswap and concurrent stress test certified.")

    finally:
        proc.kill()
        proc.wait()

if __name__ == "__main__":
    test_server_adapter_hotswap_stress()
