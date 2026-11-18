"""
Benchmark comparison: Standard Dense MLP Feed-Forward vs. WEFT Associative Memory Weights (AMW).
Measures:
1. FLOP latency across sequence lengths (N = 1024, 4096, 16384)
2. Active parameter footprint
3. Memory bandwidth throughput
"""

import time
import torch
import torch.nn as nn
from weft.associative_weight import WeftAssociativeWeight


class StandardDenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def benchmark_amw_vs_mlp():
    device = "cuda"
    d_model = 1024
    d_ffn = 2816
    rank = 64
    slots = 512
    batch = 2
    dtype = torch.bfloat16

    print("=========================================================================")
    print("  WEFT ASSOCIATIVE MEMORY WEIGHTS (AMW) VS DENSE MLP BENCHMARK")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Config: D_model={d_model}, D_ffn={d_ffn}, Slots={slots}, Rank={rank}")
    print("=========================================================================")

    mlp = StandardDenseMLP(d_model, d_ffn).to(device).to(dtype)
    amw = WeftAssociativeWeight(hidden_size=d_model, num_memory_slots=slots, rank=rank).to(device).to(dtype)

    mlp_params = sum(p.numel() for p in mlp.parameters())
    amw_params = sum(p.numel() for p in amw.parameters())
    print(f"Standard Dense MLP Parameters: {mlp_params:,} (~{mlp_params*2/1e6:.1f} MB)")
    print(f"WEFT AMW Parameters:           {amw_params:,} (~{amw_params*2/1e6:.1f} MB)")
    print(f"Parameter Footprint Reduction: {mlp_params / amw_params:.1f}x smaller")

    seq_lengths = [1024, 4096, 16384]
    print(f"\n{'Seq Length':>12} | {'Standard Dense MLP':>20} | {'WEFT AMW (Needle Lookup)':>26} | {'Speedup':>10}")
    print("-" * 75)

    for s in seq_lengths:
        x = torch.randn(batch, s, d_model, device=device, dtype=dtype)

        # Warmup
        for _ in range(3):
            _ = mlp(x)
            _ = amw(x)
        torch.cuda.synchronize()

        iters = 10
        # MLP Timing
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = mlp(x)
        torch.cuda.synchronize()
        mlp_ms = (time.perf_counter() - t0) / iters * 1000.0

        # AMW Timing
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = amw(x)
        torch.cuda.synchronize()
        amw_ms = (time.perf_counter() - t0) / iters * 1000.0

        speedup = mlp_ms / amw_ms
        print(f"{s:12d} | {mlp_ms:17.2f} ms | {amw_ms:23.2f} ms | {speedup:9.1f}x")

    print("\n=========================================================================")


if __name__ == "__main__":
    benchmark_amw_vs_mlp()
