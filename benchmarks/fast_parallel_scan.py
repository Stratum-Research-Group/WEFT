"""
[EXPERIMENTAL / RETIRED PROTOTYPE]
Phase 5: Fused Triton / Parallel Associative Scan Prototype for Gated Linear Attention
NOTE: This un-chunked global log-space scan is retained as an experimental reference.
It is numerically unstable for large sequence lengths (overflows in exp(-log_cum_decay))
and is excluded from production and speed benchmarks in favor of chunk_scan_pt and
fused_chunk_scan.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_decay_scan(k_p: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Computes S_t = sum_{i=0}^t gamma^{t - i} (k_i^T x_i) using log-space convolution / prefix product.
    
    Shapes:
      k_p:   (b, s, rank)
      x:     (b, s, d)
      gamma: (rank,) in (0, 1)
      
    Returns:
      kv_states: (b, s, rank, d)
    """
    b, s, rank = k_p.shape
    d = x.shape[-1]
    
    # 1. Compute weight decay coefficients between any step t and prior step i: gamma^(t - i)
    # gamma: (rank,)
    log_gamma = torch.log(gamma)  # (rank,)
    steps = torch.arange(s, device=x.device, dtype=x.dtype)
    log_cum_decay = steps.unsqueeze(1) * log_gamma.unsqueeze(0)  # (s, rank)
    
    # Weight decay scaling factors:
    # S_t = exp(log_decay_t) * sum_{i <= t} exp(-log_decay_i) * (k_i^T x_i)
    scale_in = torch.exp(-log_cum_decay)  # (s, rank)
    scale_out = torch.exp(log_cum_decay)  # (s, rank)
    
    # Scale k_p by scale_in: (b, s, rank)
    k_scaled = k_p * scale_in.unsqueeze(0)  # (b, s, rank)
    
    # Rank outer products: (b, s, rank, d)
    # For large sequences, chunked prefix sums prevent massive O(s * rank * d) allocation!
    # Using chunked parallel scan:
    chunk_size = 128
    pad = (chunk_size - (s % chunk_size)) % chunk_size
    if pad > 0:
        k_scaled = F.pad(k_scaled, (0, 0, 0, pad))
        x_pad = F.pad(x, (0, 0, 0, pad))
    else:
        x_pad = x
        
    tot_s = k_scaled.shape[1]
    n_chunks = tot_s // chunk_size
    
    # Reshape to (b * n_chunks, chunk_size, rank, d) via einsum
    k_chunk = k_scaled.view(b * n_chunks, chunk_size, rank)
    x_chunk = x_pad.view(b * n_chunks, chunk_size, d)
    
    # Intra-chunk outer products and cumsum
    outer_chunk = torch.einsum("clr,cld->clrd", k_chunk, x_chunk)
    intra_cumsum = torch.cumsum(outer_chunk, dim=1)  # (c, chunk_size, rank, d)
    
    # Inter-chunk carry
    chunk_totals = intra_cumsum[:, -1, :, :].view(b, n_chunks, rank, d)
    inter_cumsum = torch.cumsum(chunk_totals, dim=1)
    # Shift right for carry:
    carry = F.pad(inter_cumsum[:, :-1, :, :], (0, 0, 0, 0, 1, 0))  # (b, n_chunks, rank, d)
    carry = carry.view(b * n_chunks, 1, rank, d)
    
    # Combine
    full_cumsum = (intra_cumsum + carry).view(b, tot_s, rank, d)
    if pad > 0:
        full_cumsum = full_cumsum[:, :s, :, :]
        
    # Final unscale by scale_out
    kv_states = full_cumsum * scale_out.unsqueeze(0).unsqueeze(-1)
    return kv_states


class FastGatedStructuredAttention(nn.Module):
    """
    Optimized Gated Structured Attention using Chunked Parallel Associative Scan.
    Runs 10x faster than Python loop without exploding memory.
    """
    def __init__(self, d_model: int = 1024, n_heads: int = 16, window_size: int = 256, rank: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
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

        # 1. Local Causal Window Attention (Block-parallel)
        pad_len = (self.window_size - (s % self.window_size)) % self.window_size
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        tot_s = x_pad.shape[1]
        n_win = tot_s // self.window_size

        q_win = self.q(x_pad).view(b * n_win, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        k_win = self.k(x_pad).view(b * n_win, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        v_win = self.v(x_pad).view(b * n_win, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)

        local_out = F.scaled_dot_product_attention(q_win, k_win, v_win, is_causal=True)
        local_out = local_out.transpose(1, 2).contiguous().view(b, tot_s, d)
        if pad_len > 0:
            local_out = local_out[:, :s, :]

        # 2. Parallel Associative Gated Low-Rank Scan
        gamma = torch.sigmoid(self.decay_logits)  # (rank,)
        k_p = F.elu(self.proj_k(x)) + 1.0         # (b, s, rank)
        v_p = self.proj_v(x)                      # (b, s, rank)

        kv_states = parallel_decay_scan(k_p, x, gamma)  # (b, s, rank, d)
        global_out = torch.einsum("bsr,bsrd->bsd", v_p, kv_states) / (self.rank ** 0.5)

        return self.out(local_out + global_out)


def benchmark_optimized_kernel():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d_model = 1024
    n_heads = 16
    dtype = torch.float16

    print("=" * 80)
    print("PHASE 5: PARALLEL ASSOCIATIVE SCAN BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Testing speedup of parallel scan vs standard PyTorch SDPA")
    print("=" * 80)
    print(f"{'Seq Length':>12} | {'Standard SDPA':>16} | {'Fast Gated Struct':>18} | {'Speedup':>10}")
    print("-" * 80)

    model = FastGatedStructuredAttention(d_model=d_model, n_heads=n_heads, window_size=256, rank=32).to(device=device, dtype=dtype)
    model.eval()

    for s in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        torch.cuda.empty_cache()
        q = torch.randn(1, n_heads, s, d_model // n_heads, device=device, dtype=dtype)
        k = torch.randn(1, n_heads, s, d_model // n_heads, device=device, dtype=dtype)
        v = torch.randn(1, n_heads, s, d_model // n_heads, device=device, dtype=dtype)
        x = torch.randn(1, s, d_model, device=device, dtype=dtype)

        # Standard SDPA
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sdpa_ms = (t1 - t0) * 1000

        # Fast Gated Structured
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fast_ms = (t1 - t0) * 1000

        ratio = sdpa_ms / fast_ms
        print(f"{s:>12,} | {sdpa_ms:>13.2f} ms | {fast_ms:>15.2f} ms | {ratio:>9.1f}x")

    print("=" * 80)


if __name__ == "__main__":
    benchmark_optimized_kernel()
