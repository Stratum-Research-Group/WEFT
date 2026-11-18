"""
Comprehensive Head-to-Head Ramp-Up Benchmark on RTX 5060 Ti
Comparing Standard FlashAttention / SDPA vs Production WeftAttention
Ramping up token lengths: 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M+
"""

import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from weft import WeftConfig
from weft.modeling_weft import WeftAttention


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA device required for ramp-up benchmark.")
        return

    d_model = 1024
    n_heads = 16
    dtype = torch.float16

    print("=" * 105)
    print("PROPER RAMP-UP BENCHMARK: CAUSAL FLASHATTENTION (SDPA) vs PRODUCTION WEFT ATTENTION")
    print(f"Device: {torch.cuda.get_device_name(0)} | Free VRAM: ~{torch.cuda.mem_get_info()[0]/1024**3:.1f} GB | Precision: {dtype}")
    print(f"Config: d_model={d_model}, n_heads={n_heads}, batch_size=1, window=256, rank=16 | SDPA: Backend=FLASH_ATTENTION, is_causal=True")
    print("=" * 105)
    print(f"{'Sequence N':>12} | {'SDPA (Causal)':>14} | {'SDPA Peak VRAM':>15} | {'WEFT Latency':>15} | {'WEFT Peak VRAM':>17} | {'Speedup':>10}")
    print("-" * 105)

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

    weft_cfg = WeftConfig(
        hidden_size=d_model,
        num_attention_heads=n_heads,
        num_state_heads=4,
        state_rank=16,
        window_size=256,
    )
    weft_layer = WeftAttention(weft_cfg, layer_idx=0).to(device=device, dtype=dtype)
    weft_layer.eval()

    sdpa_alive = True

    for seq_len in seq_lengths:
        # 1. Benchmark Standard SDPA
        sdpa_time_str = "OOM / Skipped"
        sdpa_mem_str = "—"
        sdpa_ms = None

        if sdpa_alive:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                q = torch.randn(1, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)
                k = torch.randn(1, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)
                v = torch.randn(1, n_heads, seq_len, d_model // n_heads, device=device, dtype=dtype)

                iters = 10 if seq_len <= 32768 else (3 if seq_len <= 262144 else 1)
                
                # Warmup with explicit causal FlashAttention
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    for _ in range(iters):
                        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                sdpa_ms = (t1 - t0) / iters * 1000
                sdpa_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                if sdpa_ms < 1000:
                    sdpa_time_str = f"{sdpa_ms:.2f} ms"
                else:
                    sdpa_time_str = f"{sdpa_ms/1000:.2f} s"
                sdpa_mem_str = f"{sdpa_mem:.1f} MB"

                del q, k, v
            except torch.cuda.OutOfMemoryError:
                sdpa_alive = False
                sdpa_time_str = "CUDA OOM"
                sdpa_mem_str = "OOM"
            except RuntimeError as e:
                sdpa_alive = False
                sdpa_time_str = "RuntimeError"
                sdpa_mem_str = "—"

        # 2. Benchmark Production WeftAttention
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
            iters = 10 if seq_len <= 32768 else (3 if seq_len <= 262144 else 1)
            
            with torch.no_grad():
                _ = weft_layer(x)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = weft_layer(x)
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
                speedup_str = "SDPA OOM"

            del x
        except torch.cuda.OutOfMemoryError:
            st_time_str = "CUDA OOM"
            st_mem_str = "OOM"
            speedup_str = "—"
        except RuntimeError as e:
            st_time_str = "RuntimeError"
            st_mem_str = "—"
            speedup_str = "—"

        print(f"{seq_len:>12,} | {sdpa_time_str:>14} | {sdpa_mem_str:>15} | {st_time_str:>15} | {st_mem_str:>17} | {speedup_str:>10}")

    print("=" * 105)

if __name__ == "__main__":
    main()
