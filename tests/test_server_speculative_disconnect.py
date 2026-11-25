import pytest
import subprocess
import time
import socket
import os
import signal

def test_server_speculative_stream_disconnect_cleanly():
    model_path = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
    if not os.path.exists(model_path):
        pytest.skip(f"Model path {model_path} not found")

    port = 8122
    cmd = ["./build/weft-server", model_path, str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.0)
    try:
        # Connect raw socket and send request
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        req = (
            "POST /v1/speculative/verify_tree_stream HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 46\r\n\r\n"
            "{\"draft_tokens\":[1,2,3],\"target_tokens\":[1,2,3]}"
        )
        s.sendall(req.encode())
        
        # Read initial bytes then immediately close/disconnect client to trigger SIGPIPE / broken pipe on server
        _ = s.recv(64)
        s.close()
        
        # Server must still be alive and responsive to subsequent queries
        time.sleep(0.5)
        assert proc.poll() is None, "Server died on unexpected client disconnect!"
        
        # Verify subsequent request succeeds
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.connect(("127.0.0.1", port))
        s2.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        resp = s2.recv(1024).decode()
        assert "healthy" in resp
        s2.close()
        print("[+] /v1/speculative/verify_tree_stream client disconnect handled cleanly without crash.")
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2.0)
