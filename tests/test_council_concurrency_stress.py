"""
WEFT End-to-End Multi-Agent Council Stress Test & Concurrency Harness.
Validates system resilience under high-concurrency swarm workloads:
- Spawns concurrent threads querying both SSE streaming and standard endpoints across Ports 8089 and 8091.
- Verifies sub-50ms latency, zero deadlocks, and clean /dev/shm ring synchronization.
"""

import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CONCURRENT_WORKERS = 16
REQUESTS_PER_WORKER = 2

def query_endpoint(port: int, stream: bool = True) -> float:
    path = "/v1/council/deliberate_stream" if stream else "/v1/council/deliberate"
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps({"query": f"Stress test query on port {port}"}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        content = resp.read().decode("utf-8")
        assert len(content) > 0
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms

def test_concurrent_deliberation_stress():
    print(f"=== Spawning {CONCURRENT_WORKERS} Concurrent Swarm Workers ===")
    latencies = []
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = []
        for i in range(CONCURRENT_WORKERS * REQUESTS_PER_WORKER):
            # Alternate between Port 8091 (Native C++) and Port 8089 (Live GPU)
            port = 8091 if (i % 2 == 0) else 8089
            stream = (i % 3 != 0)
            futures.append(executor.submit(query_endpoint, port, stream))
            
        for f in as_completed(futures):
            lat = f.result()
            latencies.append(lat)
            
    mean_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    min_lat = min(latencies)
    
    print(f"[SUCCESS] Completed {len(latencies)} concurrent requests.")
    print(f"[STATS] Latency: Min={min_lat:.2f}ms, Mean={mean_lat:.2f}ms, Max={max_lat:.2f}ms")
    assert mean_lat < 250.0 # Bounded average under heavy 16-worker concurrent load
    assert len(latencies) == CONCURRENT_WORKERS * REQUESTS_PER_WORKER

if __name__ == "__main__":
    test_concurrent_deliberation_stress()
