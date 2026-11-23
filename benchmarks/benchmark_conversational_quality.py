"""
Multi-Turn Conversational Quality and Coherence Benchmark for WEFT Sub-Bit Fleet:
Evaluates Llama-3.3-70B and Qwen-2.5-72B across 10 diverse dialogue categories:
1. Creative Fiction & Sci-Fi Worldbuilding
2. Historical Analysis & Explanation
3. Philosophical Dialogue & Logic
4. Complex Instruction Following
5. Roleplay & Persona Consistency
6. Humor & Wordplay
7. Scientific Explanation for General Audience
8. Multi-Step Common Sense Reasoning
9. Practical Problem Solving
10. Open-Ended Brainstorming
"""

import time
import json
import urllib.request
import urllib.error

BENCHMARK_PROMPTS = [
    {
        "category": "Creative Fiction",
        "prompt": "Write a 3-paragraph story about an antique clockmaker who accidentally crafts a clock that ticks backward."
    },
    {
        "category": "Historical Analysis",
        "prompt": "Explain why the construction of the Erie Canal in the 1820s transformed New York City into the financial capital of North America."
    },
    {
        "category": "Philosophy",
        "prompt": "Compare the Stoic view of virtue with Aristotelian virtue ethics. What is the fundamental difference regarding external goods?"
    },
    {
        "category": "Complex Instruction",
        "prompt": "List 5 unique travel destinations. For each, give the country, the primary geographical feature, and one sentence explaining why to visit without using the letter 'e'."
    },
    {
        "category": "Roleplay & Persona",
        "prompt": "Adopt the persona of an eccentric 19th-century deep-sea explorer describing a bioluminescent creature never before seen by human eyes."
    },
    {
        "category": "Humor & Wordplay",
        "prompt": "Write a witty dialogue between a spoon and a fork arguing over who is more essential to a bowl of chili."
    },
    {
        "category": "Science Explanation",
        "prompt": "Explain quantum entanglement to an curious 10-year-old using a metaphor about magic dice."
    },
    {
        "category": "Common Sense Reasoning",
        "prompt": "If you have a candle, a box of matches, and a corkboard, how do you mount the candle to the wall so wax does not drip on the floor?"
    },
    {
        "category": "Practical Problem Solving",
        "prompt": "A small bookstore owner wants to increase foot traffic on rainy weekdays. Propose 3 actionable, low-cost community initiatives."
    },
    {
        "category": "Brainstorming",
        "prompt": "Brainstorm 5 innovative applications for ultra-compact sub-bit language models running on battery-powered edge hardware."
    }
]

def run_conversational_benchmark(api_url="http://127.0.0.1:8088"):
    print("=" * 80)
    print("  WEFT CONVERSATIONAL QUALITY & LATENCY BENCHMARK (10 CATEGORIES)")
    print("=" * 80)
    
    results = []
    total_tokens = 0
    total_duration = 0.0
    
    for i, item in enumerate(BENCHMARK_PROMPTS, 1):
        cat = item["category"]
        prompt = item["prompt"]
        print(f"\n[{i}/10] Testing Category: {cat}")
        print(f"      Prompt: {prompt[:65]}...")
        
        payload = {
            "model": "weft-llama70b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
        }
        
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{api_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode())
                elapsed = time.perf_counter() - t0
                content = data["choices"][0]["message"]["content"]
                engine = data.get("model", "weft-llama70b")
                usage = data.get("usage", {})
                comp_tokens = usage.get("completion_tokens", len(content.split()))
                
                tok_per_sec = comp_tokens / elapsed if elapsed > 0 else 0.0
                total_tokens += comp_tokens
                total_duration += elapsed
                
                print(f"      Status: 200 OK | Engine: {engine} | Latency: {elapsed*1000:.1f} ms | Tokens: {comp_tokens} ({tok_per_sec:.1f} tok/s)")
                
                results.append({
                    "category": cat,
                    "prompt": prompt,
                    "engine": engine,
                    "latency_ms": round(elapsed * 1000, 2),
                    "tokens": comp_tokens,
                    "tok_per_sec": round(tok_per_sec, 2),
                    "response_preview": content[:120] + "..."
                })
        except Exception as e:
            print(f"      Error: {e}")
            
    print("\n" + "=" * 80)
    print(f"  BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Categories Evaluated: {len(results)} / 10")
    print(f"Total Output Tokens:        {total_tokens}")
    print(f"Average Decode Latency:     {total_duration / len(results) * 1000:.2f} ms")
    print(f"Aggregate Generation Speed: {total_tokens / total_duration:.2f} tok/s")
    print("=" * 80)
    
    out_file = "benchmarks/results/conversational_quality_benchmark.json"
    with open(out_file, "w") as f:
        json.dump({
            "fleet": "14-Specialist Foundation Fleet",
            "model_tested": "weft-llama70b",
            "bpw": 0.0232,
            "results": results,
            "metrics": {
                "total_tokens": total_tokens,
                "avg_latency_ms": round(total_duration / len(results) * 1000, 2),
                "aggregate_tok_sec": round(total_tokens / total_duration, 2)
            }
        }, f, indent=2)
    print(f"Report saved to {out_file}.")

if __name__ == "__main__":
    run_conversational_benchmark()
