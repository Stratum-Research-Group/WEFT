"""
Phase 6: Autoregressive Streaming Inference Engine (O(1) Step Latency & 65 KB KV-Memory)
Comparing generation of 2,000 tokens between:
  1. Standard Transformer KV-Cache (grows quadratically O(T) in memory and O(T) compute per step)
  2. Weft Chunk-Gated Recurrent State (O(1) constant 65 KB memory and O(1) compute per step)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StandardTransformerKVCache(nn.Module):
    """
    Standard autoregressive generation with KV-Cache.
    At step t, stores all previous K and V vectors: size (b, n_heads, t, head_dim).
    Compute per token: O(t * d). Memory: O(t * d).
    """
    def __init__(self, d_model: int = 1024, n_heads: int = 16, vocab_size: int = 32000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_step(self, token: torch.Tensor, past_k: torch.Tensor = None, past_v: torch.Tensor = None):
        # token: (b, 1)
        x = self.embed(token)  # (b, 1, d)
        b = token.shape[0]

        q = self.q(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, 1, self.d_model)
        logits = self.head(self.out(attn_out))
        return logits, k, v


class WeftStreamingEngine(nn.Module):
    """
    Weft streaming generation.
    Maintains:
      - Local circular buffer of size window_size: O(w * d)
      - Compact recurrent global state matrix S: size (b, rank, d) -> exactly 65 KB!
    Compute per token: O(1) constant time!
    Memory: O(1) constant footprint!
    """
    def __init__(self, d_model: int = 1024, n_heads: int = 16, window_size: int = 64, rank: int = 32, vocab_size: int = 32000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.rank = rank

        self.embed = nn.Embedding(vocab_size, d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.proj_k = nn.Linear(d_model, rank, bias=False)
        self.proj_v = nn.Linear(d_model, rank, bias=False)
        self.decay_logits = nn.Parameter(torch.ones(rank) * 2.5)

        self.out = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_step(self, token: torch.Tensor, buffer_k: torch.Tensor, buffer_v: torch.Tensor, state_S: torch.Tensor):
        # token: (b, 1)
        x = self.embed(token)  # (b, 1, d)
        b = token.shape[0]

        # 1. Local window attention (bounded by circular buffer of window_size)
        q = self.q(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k_cur = self.k(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_cur = self.v(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)

        if buffer_k is not None:
            k_win = torch.cat([buffer_k, k_cur], dim=2)
            v_win = torch.cat([buffer_v, v_cur], dim=2)
            if k_win.shape[2] > self.window_size:
                k_win = k_win[:, :, -self.window_size:, :]
                v_win = v_win[:, :, -self.window_size:, :]
        else:
            k_win = k_cur
            v_win = v_cur

        local_out = F.scaled_dot_product_attention(q, k_win, v_win)
        local_out = local_out.transpose(1, 2).contiguous().view(b, 1, self.d_model)

        # 2. Global recurrent state update: O(1)
        gamma = torch.sigmoid(self.decay_logits).view(1, self.rank, 1)  # (1, rank, 1)
        k_p = (F.elu(self.proj_k(x)) + 1.0).view(b, self.rank, 1)       # (b, rank, 1)
        v_p = self.proj_v(x).view(b, 1, self.rank)                      # (b, 1, rank)

        # Query state
        global_out = torch.bmm(v_p, state_S) / (self.rank ** 0.5)        # (b, 1, d)

        # Update state: S = gamma * S + (k_p @ x)
        new_outer = torch.bmm(k_p, x)                                   # (b, rank, d)
        new_S = gamma * state_S + new_outer

        logits = self.head(self.out(local_out + global_out))
        return logits, k_win, v_win, new_S


def benchmark_generation(tokens_to_gen: int = 4000):
    dtype = torch.float16
    d_model = 1024
    n_heads = 16
    rank = 32
    window_size = 64
    vocab_size = 32000

    print("=" * 80)
    print("PHASE 6: AUTOREGRESSIVE GENERATION BENCHMARK (TOKEN-BY-TOKEN)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Generating {tokens_to_gen:,} tokens autoregressively (d_model={d_model}, rank={rank})")
    print("=" * 80)

    # 1. Benchmark Standard Transformer with KV-Cache
    std_model = StandardTransformerKVCache(d_model, n_heads, vocab_size).to(device=device, dtype=dtype)
    std_model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    curr_token = torch.tensor([[1]], device=device)
    past_k, past_v = None, None

    # Measure latency across intervals
    print(f"{'Generated Tokens':>16} | {'SDPA KV Latency (ms)':>22} | {'SDPA VRAM':>12} | {'Weft Latency (ms)':>20} | {'Weft VRAM':>12}")
    print("-" * 90)

    # Weft Model setup
    weft_model = WeftStreamingEngine(d_model, n_heads, window_size, rank, vocab_size).to(device=device, dtype=dtype)
    weft_model.eval()

    weft_token = torch.tensor([[1]], device=device)
    weft_buf_k, weft_buf_v = None, None
    state_S = torch.zeros(1, rank, d_model, device=device, dtype=dtype)

    step_checkpoints = [100, 500, 1000, 2000, 3000, 4000]
    std_times = {}
    weft_times = {}

    # Run Standard KV generation
    for step in range(1, tokens_to_gen + 1):
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, past_k, past_v = std_model.forward_step(curr_token, past_k, past_v)
            curr_token = logits.argmax(dim=-1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step in step_checkpoints:
            std_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            std_times[step] = ((t1 - t0) * 1000, std_mem)

    # Run Weft generation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, tokens_to_gen + 1):
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, weft_buf_k, weft_buf_v, state_S = weft_model.forward_step(weft_token, weft_buf_k, weft_buf_v, state_S)
            weft_token = logits.argmax(dim=-1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step in step_checkpoints:
            weft_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            weft_times[step] = ((t1 - t0) * 1000, weft_mem)

    for step in step_checkpoints:
        s_lat, s_mem = std_times[step]
        w_lat, w_mem = weft_times[step]
        print(f"{step:>16,} | {s_lat:>20.2f} ms | {s_mem:>10.1f} MB | {w_lat:>18.2f} ms | {w_mem:>10.1f} MB")

    print("=" * 90)


if __name__ == "__main__":
    benchmark_generation()
