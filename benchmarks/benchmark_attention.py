"""
Benchmark comparing standard PyTorch SDPA (FlashAttention / cuDNN)
against production WeftAttention (Sliding-Window + Gated Low-Rank Recurrence)
as well as the early prototype on GPU.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from weft import WeftConfig
from weft.modeling_weft import WeftAttention


class StructuredAttention(nn.Module):
    """
    Hypothesis A Prototype: Block-Band + Low-Rank Projected Attention.
    
    Instead of full N x N dense pairwise attention:
      Output = LocalBandAttention(Q, K, V; window_size) + LowRankAttention(Q, K, V; rank_k)
    where:
      - LocalBandAttention: Sliding window attention of size w (O(N * w))
      - LowRankAttention: Q @ (E_K^T @ E_V) via projected rank-k bottleneck (O(N * k * d))
    """
    def __init__(self, d_model: int, n_heads: int, window_size: int = 256, rank: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.rank = rank

        # Linear projections for low-rank bottleneck
        self.proj_k = nn.Linear(d_model, rank, bias=False)
        self.proj_v = nn.Linear(d_model, rank, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d = x.shape
        # 1. Local band attention (chunked / windowed)
        # For benchmarking compute/memory bounds, partition sequence into non-overlapping windows of window_size
        pad_len = (self.window_size - (seq_len % self.window_size)) % self.window_size
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        
        total_len = x_pad.shape[1]
        n_windows = total_len // self.window_size
        
        # Reshape to (b * n_windows, window_size, d)
        x_win = x_pad.view(b * n_windows, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        # Local window SDPA
        local_out = F.scaled_dot_product_attention(x_win, x_win, x_win)
        local_out = local_out.transpose(1, 2).contiguous().view(b, total_len, d)
        if pad_len > 0:
            local_out = local_out[:, :seq_len, :]

        # 2. Global low-rank path
        # Compress K and V through rank bottleneck: (b, rank, d)
        k_comp = torch.bmm(self.proj_k(x).transpose(1, 2), x) / seq_len  # (b, rank, d)
        v_comp = self.proj_v(x)  # (b, seq_len, rank)
        
        # Global context vector:
        global_out = torch.bmm(v_comp, k_comp)  # (b, seq_len, d)
        
        return self.out_proj(local_out + global_out)


def benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = torch.device("cuda:0")
    d_model = 1024
    n_heads = 16
    batch_size = 1
    dtype = torch.float16

    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    struct_layer = StructuredAttention(d_model=d_model, n_heads=n_heads, window_size=256, rank=32).to(device=device, dtype=dtype)
    struct_layer.eval()

    weft_cfg = WeftConfig(
        hidden_size=d_model,
        num_attention_heads=n_heads,
        num_state_heads=4,
        state_rank=16,
        window_size=256,
    )
    weft_layer = WeftAttention(weft_cfg, layer_idx=0).to(device=device, dtype=dtype)
    weft_layer.eval()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: d_model={d_model}, n_heads={n_heads}, batch_size={batch_size}, dtype={dtype}")
    print(f"{'Seq Length':>10} | {'SDPA (ms)':>10} | {'SDPA Mem':>10} | {'Struct (ms)':>12} | {'Struct Mem':>12} | {'WEFT (ms)':>10} | {'WEFT Mem':>10}")
    print("-" * 90)

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        q = torch.randn(batch_size, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        # Benchmark SDPA
        sdpa_time = None
        sdpa_mem = None
        try:
            # Warmup
            for _ in range(5):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            iters = 20
            for _ in range(iters):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            sdpa_time = (t1 - t0) / iters * 1000
            sdpa_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            sdpa_time = -1
            sdpa_mem = -1

        # Benchmark StructuredAttention
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        struct_time = None
        struct_mem = None
        try:
            with torch.no_grad():
                for _ in range(5):
                    _ = struct_layer(x)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                iters = 20
                for _ in range(iters):
                    _ = struct_layer(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                struct_time = (t1 - t0) / iters * 1000
                struct_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            struct_time = -1
            struct_mem = -1

        # Benchmark Production WeftAttention
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weft_time = None
        weft_mem = None
        try:
            with torch.no_grad():
                for _ in range(5):
                    _ = weft_layer(x)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                iters = 20
                for _ in range(iters):
                    _ = weft_layer(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                weft_time = (t1 - t0) / iters * 1000
                weft_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            weft_time = -1
            weft_mem = -1

        s_time_str = f"{sdpa_time:.2f}" if sdpa_time != -1 else "OOM"
        s_mem_str = f"{sdpa_mem:.1f} MB" if sdpa_mem != -1 else "OOM"
        st_time_str = f"{struct_time:.2f}" if struct_time != -1 else "OOM"
        st_mem_str = f"{struct_mem:.1f} MB" if struct_mem != -1 else "OOM"
        w_time_str = f"{weft_time:.2f}" if weft_time != -1 else "OOM"
        w_mem_str = f"{weft_mem:.1f} MB" if weft_mem != -1 else "OOM"

        print(f"{seq_len:>10} | {s_time_str:>10} | {s_mem_str:>10} | {st_time_str:>12} | {st_mem_str:>12} | {w_time_str:>10} | {w_mem_str:>10}")


if __name__ == "__main__":
    benchmark()
