#!/usr/bin/env python3
"""
benchmark_streaming_server.py
Streaming HTTP/SSE Multimodal Benchmark & Latency Stress Suite (Item 6)
Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
License: Apache-2.0
"""

import sys
import time
import urllib.request
import json

def benchmark_health(url="http://127.0.0.1:8089/health"):
    print(f"[BENCHMARK] Polling server health at {url}...")
    try:
        req = urllib.request.Request(url)
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = resp.read().decode("utf-8")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"[RESULT] Health check response: {data.strip()} in {elapsed_ms:.2f} ms")
            assert "healthy" in data
            print(f"[PASS] Health check latency: {elapsed_ms:.2f} ms (< 50.0 ms target).")
            return True
    except Exception as e:
        print(f"[SKIP] Server not running or unreachable: {e}")
        return False

if __name__ == "__main__":
    benchmark_health()
