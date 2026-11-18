"""
Head-to-head empirical pretraining comparison:
Standard WEFT (Dense MLP) vs. WEFT-AMW (Associative Memory Weights)
Trained with Muon low-memory optimizer on live FineWeb-Edu stream.
Measures:
1. Training loss convergence rate
2. Tokens/second throughput
3. Peak GPU memory footprint
"""

import time
import torch
from weft import WeftConfig, WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer
from weft.optimizers import Muon


def run_head_to_head_comparison(steps: int = 40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    seq_len = 128
    hidden_size = 256
    layers = 4
    heads = 4

    print("=========================================================================")
    print("  WEFT HEAD-TO-HEAD: DENSE MLP VS. ASSOCIATIVE MEMORY WEIGHTS (AMW)")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Config: D_model={hidden_size}, Layers={layers}, Heads={heads}, Steps={steps}")
    print("=========================================================================")

    # 1. Models instantiation
    cfg_dense = WeftConfig(
        vocab_size=50257,
        hidden_size=hidden_size,
        intermediate_size=768,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_state_heads=2,
        state_rank=64,
        window_size=64,
        use_amw=False,
    )
    model_dense = WeftForCausalLM(cfg_dense).to(device)

    cfg_amw = WeftConfig(
        vocab_size=50257,
        hidden_size=hidden_size,
        intermediate_size=768,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_state_heads=2,
        state_rank=64,
        window_size=64,
        use_amw=True,
        amw_num_slots=256,
        amw_rank=64,
    )
    model_amw = WeftForCausalLM(cfg_amw).to(device)

    params_dense = sum(p.numel() for p in model_dense.parameters())
    params_amw = sum(p.numel() for p in model_amw.parameters())

    print(f"Standard Dense Model: {params_dense:,} parameters (~{params_dense*2/1e6:.1f} MB)")
    print(f"WEFT-AMW Model:       {params_amw:,} parameters (~{params_amw*2/1e6:.1f} MB)")
    print(f"Parameter Savings:    {params_dense - params_amw:,} parameters ({100*(1 - params_amw/params_dense):.1f}% reduction)")

    # 2. Data streamer
    streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=str(device))
    batches = [streamer.get_batch() for _ in range(steps)]

    # 3. Train Standard Dense Model with Muon
    opt_dense = Muon(model_dense.parameters(), lr=0.01)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    losses_dense = []
    for step in range(steps):
        model_dense.train()
        x, y = batches[step]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model_dense(input_ids=x, labels=y).loss
        opt_dense.zero_grad(set_to_none=True)
        loss.backward()
        opt_dense.step()
        losses_dense.append(loss.item())

    torch.cuda.synchronize()
    t_dense = time.perf_counter() - t0
    dense_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    tok_s_dense = (steps * batch_size * seq_len) / t_dense

    # 4. Train WEFT-AMW Model with Muon
    opt_amw = Muon(model_amw.parameters(), lr=0.01)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    losses_amw = []
    for step in range(steps):
        model_amw.train()
        x, y = batches[step]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model_amw(input_ids=x, labels=y).loss
        opt_amw.zero_grad(set_to_none=True)
        loss.backward()
        opt_amw.step()
        losses_amw.append(loss.item())

    torch.cuda.synchronize()
    t_amw = time.perf_counter() - t0
    amw_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    tok_s_amw = (steps * batch_size * seq_len) / t_amw

    # Results table
    print("\n=========================================================================")
    print("  HEAD-TO-HEAD TRAINING BENCHMARK RESULTS")
    print("=========================================================================")
    print(f"{'Metric':<28} | {'Standard Dense WEFT':<20} | {'WEFT-AMW (Associative)':<22}")
    print("-" * 75)
    print(f"{'Total Parameters':<28} | {params_dense:>17,}  | {params_amw:>19,} ")
    print(f"{'Peak VRAM (Weights+Opt)':<28} | {dense_vram:>17.1f} MB | {amw_vram:>19.1f} MB")
    print(f"{'Training Time (40 steps)':<28} | {t_dense:>17.2f} s  | {t_amw:>19.2f} s ")
    print(f"{'Training Throughput':<28} | {tok_s_dense:>15.1f} t/s | {tok_s_amw:>17.1f} t/s")
    print(f"{'Initial Loss (Step 1)':<28} | {losses_dense[0]:>17.4f}  | {losses_amw[0]:>19.4f} ")
    print(f"{'Final Loss (Step 40)':<28} | {losses_dense[-1]:>17.4f}  | {losses_amw[-1]:>19.4f} ")
    print("-" * 75)
    print(f"Throughput Speedup: {tok_s_amw / tok_s_dense:.2f}x faster training")
    print("=========================================================================")


if __name__ == "__main__":
    run_head_to_head_comparison()
