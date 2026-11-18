"""
Comprehensive Head-to-Head Ramp-Up Benchmark on RTX 5060 Ti
Comparing Standard FlashAttention / SDPA vs Structured Linear Attention
Ramping up token lengths: 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M+
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int = 256, rank: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.rank = rank

        self.proj_k = nn.Linear(d_model, rank, bias=False)
        self.proj_v = nn.Linear(d_model, rank, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d = x.shape
        pad_len = (self.window_size - (seq_len % self.window_size)) % self.window_size
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        
        total_len = x_pad.shape[1]
        n_windows = total_len // self.window_size
        
        # 1. Local block attention
        x_win = x_pad.view(b * n_windows, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        local_out = F.scaled_dot_product_attention(x_win, x_win, x_win)
        local_out = local_out.transpose(1, 2).contiguous().view(b, total_len, d)
        if pad_len > 0:
            local_out = local_out[:, :seq_len, :]

        # 2. Global rank-32 bottleneck
        k_comp = torch.bmm(self.proj_k(x).transpose(1, 2), x) / seq_len
        v_comp = self.proj_v(x)
        global_out = torch.bmm(v_comp, k_comp)
        
        return self.out_proj(local_out + global_out)

def main():
    device = torch.device("cuda:0")
    d_model = 1024
    n_heads = 16
    dtype = torch.float16

    print("=" * 95)
    print("PROPER RAMP-UP BENCHMARK: STANDARD SDPA vs STRUCTURED ATTENTION")
    print(f"Device: {torch.cuda.get_device_name(0)} | Free VRAM: ~13.7 GB | Precision: {dtype}")
    print(f"Config: d_model={d_model}, n_heads={n_heads}, batch_size=1, window=256, rank=32")
    print("=" * 95)
    print(f"{'Sequence N':>12} | {'SDPA Latency':>14} | {'SDPA Peak VRAM':>15} | {'Struct Latency':>15} | {'Struct Peak VRAM':>17} | {'Speedup':>10}")
    print("-" * 95)

    seq_lengths = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
    ]

    struct_layer = StructuredAttention(d_model=d_model, n_heads=n_heads, window_size=256, rank=32).to(device=device, dtype=dtype)
    struct_layer.eval()

    sdpa_alive = True

    for seq_len in seq_lengths:
        # 1. Benchmark Standard SDPA
        sdpa_time_str = "OOM / Skipped"
        sdpa_mem_str = "—"
        sdpa_ms = None

        if sdpa_alive:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                q = torch.randn(1, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)
                k = torch.randn(1, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)
                v = torch.randn(1, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)

                # Iterations: fewer for large N to avoid long waits
                iters = 10 if seq_len <= 32768 else (3 if seq_len <= 262144 else 1)
                
                # Warmup
                _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                sdpa_ms = (t1 - t0) / iters * 1000
                sdpa_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                if sdpa_ms < 1000:
                    sdpa_time_str = f"{sdpa_ms:.2f} ms"
                else:
                    sdpa_time_str = f"{sdpa_ms/1000:.2f} s"
                sdpa_mem_str = f"{sdpa_mem:.1f} MB"
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                sdpa_alive = False
                sdpa_time_str = "CUDA OOM"
                sdpa_mem_str = ">13.7 GB"

        # 2. Benchmark Structured Attention
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
            iters = 10 if seq_len <= 32768 else (3 if seq_len <= 262144 else 1)
            
            with torch.no_grad():
                _ = struct_layer(x)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = struct_layer(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

            st_ms = (t1 - t0) / iters * 1000
            st_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if st_ms < 1000:
                st_time_str = f"{st_ms:.2f} ms"
            else:
                st_time_str = f"{st_ms/1000:.2f} s"
            st_mem_str = f"{st_mem:.1f} MB"

            if sdpa_ms is not None:
                ratio = sdpa_ms / st_ms
                speedup_str = f"{ratio:.1f}x"
            else:
                speedup_str = "Inf (SDPA OOM)"
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            st_time_str = "CUDA OOM"
            st_mem_str = ">13.7 GB"
            speedup_str = "—"

        print(f"{seq_len:>12,} | {sdpa_time_str:>14} | {sdpa_mem_str:>15} | {st_time_str:>15} | {st_mem_str:>17} | {speedup_str:>10}")

    print("=" * 95)

if __name__ == "__main__":
    main()
