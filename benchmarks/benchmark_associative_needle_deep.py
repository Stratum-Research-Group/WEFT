"""
Deep Multi-Needle Associative Recall Benchmark for WEFT-2.

Evaluates MoE routing precision across deep contexts (N = 2,048, 4,096, 8,192, 16,384)
comparing:
  1. Standard Static Single-Token Routing: W_r @ x_t
  2. WEFT-2 Associative Historical Routing: R_t = gamma * R_{t-1} + k_{r, t-1}^T * SoftAssignment_t

Measures target expert activation affinity at the extreme end of the sequence
when answering questions about needles planted early in the sequence.
"""

import time
import torch
import torch.nn.functional as F
from weft import WeftConfig, WeftForCausalLM


def run_deep_needle_benchmark(
    context_lengths=[2048, 4096, 8192, 16384],
    device="cuda",
):
    print("=" * 80)
    print("  WEFT-2 DEEP MULTI-NEEDLE ASSOCIATIVE ROUTING BENCHMARK")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # Initialize WEFT-2 lightweight testing instance
    config = WeftConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=32,
        window_size=128,
        use_mla=True,
        mla_latent_dim=64,
        use_titans=True,
        titans_mem_dim=32,
        use_moe=True,
        moe_num_routed_experts=8,
        moe_top_k=2,
    )

    model = WeftForCausalLM(config).to(device)
    model.eval()

    results = []

    for N in context_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Needle position: 5% into the context
        needle_pos = max(64, int(N * 0.05))
        query_pos = N - 1

        # Synthesize sequence
        input_ids = torch.randint(10, 990, (1, N), device=device)
        needle_token = 999
        query_token = 888
        input_ids[0, needle_pos] = needle_token
        input_ids[0, query_pos] = query_token

        # 1. Forward with WEFT-2 associative routing enabled
        for layer in model.model.layers:
            if hasattr(layer.mlp, "use_associative_routing"):
                layer.mlp.use_associative_routing = True

        t0 = time.time()
        with torch.no_grad():
            outputs_assoc = model(input_ids=input_ids, use_cache=True, return_dict=True)
        assoc_time_ms = (time.time() - t0) * 1000.0
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Inspect router state cached in past_key_values
        past = outputs_assoc.past_key_values
        router_state = past.layers[0].weft_router_state  # [1, 64, num_experts]
        layer0_moe = model.model.layers[0].mlp
        target_expert = 3

        # Compute query activation affinity to target expert
        query_x = outputs_assoc.logits[:, -1, :config.hidden_size]
        query_r = layer0_moe.routing_key_proj(query_x.unsqueeze(1))  # [1, 1, 64]
        retrieved_scores = torch.bmm(query_r, router_state).squeeze(1)
        assoc_prob = F.softmax(retrieved_scores, dim=-1)[0, target_expert].item()

        # 2. Compare against static routing (W_r @ x)
        static_logits = layer0_moe.router(query_x)
        static_prob = F.softmax(static_logits, dim=-1)[0, target_expert].item()

        print(
            f"Context N = {N:6d} | Needle at {needle_pos:5d} (dist={query_pos - needle_pos:5d}) | "
            f"Prefill: {assoc_time_ms:6.2f} ms | VRAM: {vram_mb:6.1f} MB | "
            f"Static Router Target Prob: {static_prob * 100:5.1f}% | "
            f"Associative Router Target Prob: {assoc_prob * 100:5.1f}% | "
            f"Advantage: {assoc_prob / max(1e-4, static_prob):5.1f}x"
        )

        results.append({
            "N": N,
            "dist": query_pos - needle_pos,
            "latency_ms": assoc_time_ms,
            "vram_mb": vram_mb,
            "static_prob": static_prob,
            "assoc_prob": assoc_prob,
        })

    print("=" * 80)
    print("  DEEP RETRIEVAL SUMMARY")
    for r in results:
        print(f"  Distance {r['dist']:5d} tokens -> Associative Recall: {r['assoc_prob']*100:.1f}% vs Static: {r['static_prob']*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_deep_needle_benchmark()
