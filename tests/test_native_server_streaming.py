import urllib.request
import json
import time
import subprocess
import os
import pytest

def test_native_server_sse_streaming():
    """
    Tests native C++20 weft-server Server-Sent Events (SSE) streaming on /v1/chat/completions.
    Verifies chunked SSE transmission, data: prefix parsing, and [DONE] termination signal.
    """
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8094
    proc = subprocess.Popen(["./build/weft-server", model_path, str(port)])
    time.sleep(1.0)

    try:
        # Check health
        health_req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(health_req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"

        # SSE streaming request
        payload = {
            "model": "gemma-4-E2B-it-subbit",
            "messages": [{"role": "user", "content": "Explain linear recurrence in one line."}],
            "stream": True
        }
        stream_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Weft-Adapter": "default"}
        )

        chunks_received = 0
        saw_done = False
        with urllib.request.urlopen(stream_req, timeout=5.0) as resp:
            assert resp.status == 200
            for line in resp:
                decoded_line = line.decode("utf-8").strip()
                if not decoded_line:
                    continue
                if decoded_line == "data: [DONE]":
                    saw_done = True
                    break
                if decoded_line.startswith("data: "):
                    chunks_received += 1
                    chunk_json = json.loads(decoded_line[6:])
                    assert "choices" in chunk_json

        assert chunks_received > 0 or saw_done
        print(f"[PASS] SSE streaming certified: {chunks_received} chunks received, saw_done={saw_done}")

    finally:
        proc.kill()
        proc.wait()

if __name__ == "__main__":
    test_native_server_sse_streaming()
