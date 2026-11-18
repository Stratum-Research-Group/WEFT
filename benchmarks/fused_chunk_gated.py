"""
[CHUNK-PROTOTYPE / DELAYED GLOBAL READOUT VARIANT]
Phase 5: Fused Chunk-Linear Recurrence (Lightning Attention / RetNet style prototype)
NOTE: This prototype uses a delayed chunk-level global readout where tokens in chunk c
query state from chunk c-1 rather than the exact causal intra-chunk token recurrence
used in WeftAttention. It is evaluated as a separate chunk-level architectural exploration.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedChunkGatedAttention(nn.Module):
    def __init__(self, d_model: int = 1024, n_heads: int = 16, chunk_size: int = 256, rank: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size
        self.rank = rank

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.proj_k = nn.Linear(d_model, rank, bias=False)
        self.proj_v = nn.Linear(d_model, rank, bias=False)
        self.decay_logits = nn.Parameter(torch.ones(rank) * 2.5)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        c = self.chunk_size

        pad_len = (c - (s % c)) % c
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        tot_s = x.shape[1]
        n_chunks = tot_s // c

        # 1. Local chunk causal attention
        q = self.q(x).view(b * n_chunks, c, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b * n_chunks, c, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b * n_chunks, c, self.n_heads, self.head_dim).transpose(1, 2)
        local_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        local_out = local_out.transpose(1, 2).contiguous().view(b, tot_s, d)

        # 2. Chunk-wise recurrent state S: (b, rank, d)
        gamma = torch.sigmoid(self.decay_logits)  # (rank,)
        gamma_c = gamma ** c                      # decay across a full chunk: (rank,)
        
        # Intra-chunk decay mask: (c, 1)
        steps = torch.arange(c, device=x.device, dtype=x.dtype)
        # decay_step_i = gamma^(c - 1 - i)
        decay_to_end = gamma.unsqueeze(0) ** (c - 1 - steps.unsqueeze(1))  # (c, rank)
        decay_from_start = gamma.unsqueeze(0) ** steps.unsqueeze(1)        # (c, rank)

        k_p = F.elu(self.proj_k(x)) + 1.0  # (b, tot_s, rank)
        v_p = self.proj_v(x)               # (b, tot_s, rank)

        # Chunk views: (b, n_chunks, c, rank/d)
        k_chunks = k_p.view(b, n_chunks, c, self.rank)
        v_chunks = v_p.view(b, n_chunks, c, self.rank)
        x_chunks = x.view(b, n_chunks, c, d)

        global_outs = []
        # State S is only (b, rank, d) -> 65 KB!
        state_S = torch.zeros(b, self.rank, d, device=x.device, dtype=x.dtype)

        for i in range(n_chunks):
            kc = k_chunks[:, i, :, :]  # (b, c, rank)
            vc = v_chunks[:, i, :, :]  # (b, c, rank)
            xc = x_chunks[:, i, :, :]  # (b, c, d)

            # Query historical state
            # out_hist: (b, c, rank) @ (b, rank, d) scaled by decay_from_start
            vc_scaled = vc * decay_from_start.unsqueeze(0)  # (b, c, rank)
            chunk_global_out = torch.bmm(vc_scaled, state_S)  # (b, c, d)
            global_outs.append(chunk_global_out)

            # Update state S with current chunk:
            # S = gamma^C * S + sum_{j} decay_to_end_j * (k_j^T x_j)
            kc_scaled = kc * decay_to_end.unsqueeze(0)  # (b, c, rank)
            chunk_update = torch.bmm(kc_scaled.transpose(1, 2), xc)  # (b, rank, d)
            state_S = gamma_c.view(1, self.rank, 1) * state_S + chunk_update

        global_out = torch.cat(global_outs, dim=1) / (self.rank ** 0.5)

        if pad_len > 0:
            local_out = local_out[:, :s, :]
            global_out = global_out[:, :s, :]

        return self.out(local_out + global_out)


def run_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d_model = 1024
    n_heads = 16
    dtype = torch.float16

    print("=" * 80)
    print("PHASE 5: CHUNK-RECURRENT GATED ATTENTION BENCHMARK (65 KB STATE)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print(f"{'Sequence N':>12} | {'Standard SDPA':>16} | {'Chunk-Gated':>16} | {'Speedup':>10} | {'Peak VRAM':>12}")
    print("-" * 80)

    model = FusedChunkGatedAttention(d_model=d_model, n_heads=n_heads, chunk_size=256, rank=32).to(device=device, dtype=dtype)
    model.eval()

    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

    for s in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, s, d_model, device=device, dtype=dtype)
        q = torch.randn(1, n_heads, s, d_model // n_heads, device=device, dtype=dtype)
        k = torch.randn(1, n_heads, s, d_model // n_heads, device=device, dtype=dtype)
        v = torch.randn(1, n_heads, s, d_model // n_heads, device=device, dtype=dtype)

        # Standard SDPA
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sdpa_ms = (t1 - t0) * 1000

        # Chunk-Gated
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            chunk_ms = (t1 - t0) * 1000
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        ratio = sdpa_ms / chunk_ms
        sdpa_str = f"{sdpa_ms:.2f} ms" if sdpa_ms < 1000 else f"{sdpa_ms/1000:.2f} s"
        chunk_str = f"{chunk_ms:.2f} ms" if chunk_ms < 1000 else f"{chunk_ms/1000:.2f} s"
        print(f"{s:>12,} | {sdpa_str:>16} | {chunk_str:>16} | {ratio:>9.1f}x | {peak_vram:>9.1f} MB")

    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
