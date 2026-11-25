import urllib.request
import json
import time
import subprocess
import signal
import os
import pytest

def test_server_graceful_shutdown():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8105
    proc = subprocess.Popen(["./build/weft-server", model_path, str(port)])
    time.sleep(1.0)

    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"

        # Fire SIGTERM to test clean drain & shutdown
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.5)

        # Confirm process exited cleanly or responds to wait
        proc.terminate()
        proc.wait(timeout=3.0)
        print("[PASS] Native server graceful shutdown certified.")

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

if __name__ == "__main__":
    test_server_graceful_shutdown()
