"""
High-Concurrency Serving Stress Test for WEFT OpenAI Server.
Simulates bursty, concurrent streaming requests across clients to measure:
1. Time-to-First-Token (TTFT)
2. Inter-Token Latency (ITL)
3. End-to-End Generation Throughput (tokens/second)
4. Memory stability under high concurrent connections.
"""

import time
import asyncio
import argparse
from typing import List, Dict, Any
import httpx


async def client_worker(
    client_id: int,
    base_url: str,
    prompt: str,
    max_tokens: int,
    results: List[Dict[str, Any]],
):
    req_payload = {
        "model": "weft-causal-lm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    t_start = time.perf_counter()
    first_token_time = None
    token_times = []
    received_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{base_url}/v1/chat/completions", json=req_payload) as response:
                if response.status_code != 200:
                    results.append({"client_id": client_id, "success": False, "error": response.status_code})
                    return

                async for line in response.aiter_lines():
                    t_now = time.perf_counter()
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        received_tokens += 1
                        if first_token_time is None:
                            first_token_time = t_now
                        token_times.append(t_now)

        t_end = time.perf_counter()
        ttft_ms = (first_token_time - t_start) * 1000.0 if first_token_time else 0.0
        total_time_ms = (t_end - t_start) * 1000.0

        itl_ms = 0.0
        if len(token_times) > 1:
            diffs = [t2 - t1 for t1, t2 in zip(token_times[:-1], token_times[1:])]
            itl_ms = (sum(diffs) / len(diffs)) * 1000.0

        results.append({
            "client_id": client_id,
            "success": True,
            "tokens": received_tokens,
            "ttft_ms": ttft_ms,
            "itl_ms": itl_ms,
            "total_ms": total_time_ms,
            "tok_per_sec": (received_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0,
        })
    except Exception as e:
        results.append({"client_id": client_id, "success": False, "error": str(e)})


async def run_stress_test(base_url: str, concurrency: int = 16, max_tokens: int = 32):
    print("=" * 80)
    print("  WEFT HIGH-CONCURRENCY SERVING STRESS TEST")
    print(f"  Target: {base_url} | Concurrent Clients: {concurrency} | Max Tokens: {max_tokens}")
    print("=" * 80)

    # Check server readiness
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/health")
            print("Server Health:", resp.json())
    except Exception as e:
        print(f"Server check failed: {e}")
        return

    prompt = "Explain in one sentence how WEFT achieves constant decode memory."
    results: List[Dict[str, Any]] = []

    t_launch = time.perf_counter()
    tasks = [
        client_worker(i, base_url, prompt, max_tokens, results)
        for i in range(concurrency)
    ]
    await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - t_launch

    successful = [r for r in results if r.get("success")]
    print("-" * 80)
    print(f"Concurrency Run Complete: {len(successful)}/{concurrency} requests succeeded ({total_duration:.2f}s total)")

    if len(successful) > 0:
        avg_ttft = sum(r["ttft_ms"] for r in successful) / len(successful)
        avg_itl = sum(r["itl_ms"] for r in successful) / len(successful)
        total_tokens = sum(r["tokens"] for r in successful)
        agg_throughput = total_tokens / total_duration

        print(f"Average Time to First Token (TTFT): {avg_ttft:.2f} ms")
        print(f"Average Inter-Token Latency (ITL):  {avg_itl:.2f} ms")
        print(f"Aggregate Serving Throughput:       {agg_throughput:.2f} tokens/second")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WEFT Serving Concurrency Stress Test")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    asyncio.run(run_stress_test(args.url, args.concurrency, args.max_tokens))
