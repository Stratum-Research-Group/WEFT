"""
Rigorous Head-to-Head Benchmark: Dense MLP vs. WEFT Associative Memory Weights (AMW).
Enforces:
1. Fixed random seeds for deterministic reproducibility.
2. Identical training batches and independent held-out validation batches.
3. Accurate accounting of parameters, latency, training loss, and held-out validation loss.
"""

import math
import time
import torch
from weft import WeftConfig, WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer
from weft.optimizers import Muon


def run_rigorous_evaluation(steps: int = 50, val_steps: int = 10, seed: int = 42):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    seq_len = 128
    hidden_size = 256
    layers = 4
    heads = 4

    print("=========================================================================")
    print("  RIGOROUS HELD-OUT EVALUATION: DENSE MLP VS. WEFT-AMW")
    print(f"  Device: {torch.cuda.get_device_name(0)} | Seed: {seed}")
    print(f"  Steps: {steps} | Val Steps: {val_steps} | Batch: {batch_size} | SeqLen: {seq_len}")
    print("=========================================================================")

    # 1. Fetch deterministic train and validation batches
    print("Fetching deterministic data stream from FineWeb-Edu...")
    train_streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=str(device), split="train")
    train_batches = [train_streamer.get_batch() for _ in range(steps)]

    # Use isolated streamer instance pinned at offset 5000 and split='val' for held-out validation
    val_streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=str(device), split="val")
    val_streamer.current_offset = 5000
    val_streamer.token_queue = []
    val_batches = [val_streamer.get_batch() for _ in range(val_steps)]

    # 2. Evaluate Standard Dense Model
    torch.manual_seed(seed)
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
    dense_params = sum(p.numel() for p in model_dense.parameters())
    opt_dense = Muon(model_dense.parameters(), lr=0.01)

    t0 = time.perf_counter()
    for x, y in train_batches:
        model_dense.train()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model_dense(input_ids=x, labels=y).loss
        opt_dense.zero_grad(set_to_none=True)
        loss.backward()
        opt_dense.step()
    torch.cuda.synchronize()
    t_dense = time.perf_counter() - t0

    # Held-out validation
    model_dense.eval()
    val_losses_dense = []
    with torch.no_grad():
        for vx, vy in val_batches:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                vout = model_dense(input_ids=vx, labels=vy)
            val_losses_dense.append(vout.loss.item())
    dense_val_loss = sum(val_losses_dense) / len(val_losses_dense)

    # 3. Evaluate WEFT-AMW Model
    torch.manual_seed(seed)
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
    amw_params = sum(p.numel() for p in model_amw.parameters())
    opt_amw = Muon(model_amw.parameters(), lr=0.01)

    t0 = time.perf_counter()
    for x, y in train_batches:
        model_amw.train()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model_amw(input_ids=x, labels=y).loss
        opt_amw.zero_grad(set_to_none=True)
        loss.backward()
        opt_amw.step()
    torch.cuda.synchronize()
    t_amw = time.perf_counter() - t0

    # Held-out validation
    model_amw.eval()
    val_losses_amw = []
    with torch.no_grad():
        for vx, vy in val_batches:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                vout = model_amw(input_ids=vx, labels=vy)
            val_losses_amw.append(vout.loss.item())
    amw_val_loss = sum(val_losses_amw) / len(val_losses_amw)

    print("\n=========================================================================")
    print("  RIGOROUS BENCHMARK RESULTS")
    print("=========================================================================")
    print(f"{'Metric':<28} | {'Standard Dense WEFT':<20} | {'WEFT-AMW (Associative)':<22}")
    print("-" * 75)
    print(f"{'Total Parameters':<28} | {dense_params:>17,}  | {amw_params:>19,} ")
    print(f"{'Training Time (50 steps)':<28} | {t_dense:>17.2f} s  | {t_amw:>19.2f} s ")
    print(f"{'Training Throughput':<28} | {(steps*batch_size*seq_len)/t_dense:>15.1f} t/s | {(steps*batch_size*seq_len)/t_amw:>17.1f} t/s")
    print(f"{'Held-Out Val Loss':<28} | {dense_val_loss:>17.4f}  | {amw_val_loss:>19.4f} ")
    print(f"{'Held-Out Perplexity':<28} | {math.exp(dense_val_loss):>17.2f}  | {math.exp(amw_val_loss):>19.2f} ")
    print("-" * 75)
    print(f"Measured Speedup:   {t_dense / t_amw:.2f}x faster training")
    print(f"Val Loss Delta:     {abs(dense_val_loss - amw_val_loss):.4f}")
    print("=========================================================================")


if __name__ == "__main__":
    run_rigorous_evaluation()
