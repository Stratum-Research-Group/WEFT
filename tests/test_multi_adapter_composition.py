import requests
import subprocess
import time
import os

def test_multi_adapter_composition():
    port = 8092
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    proc = subprocess.Popen(["./build/weft-server", model_path, str(port)])
    try:
        time.sleep(1)
        # Check health
        res = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        assert res.status_code == 200

        # Load two specialist adapters
        res1 = requests.post(f"http://127.0.0.1:{port}/v1/adapters/load", json={"id": "coder"}, timeout=5)
        assert res1.status_code == 200
        res2 = requests.post(f"http://127.0.0.1:{port}/v1/adapters/load", json={"id": "prover"}, timeout=5)
        assert res2.status_code == 200

        # Test composite routing header
        headers = {
            "X-Weft-Adapters": "coder=0.6,prover=0.4"
        }
        res_comp = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "synthesize"}]},
            headers=headers,
            timeout=5
        )
        assert res_comp.status_code == 200
        data = res_comp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        print("[PASS] Multi-adapter composition test verified:", data["choices"][0]["message"]["content"])
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_multi_adapter_composition()
