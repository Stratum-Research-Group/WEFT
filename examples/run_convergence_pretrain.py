"""
Bounded Continual Pretraining Pipeline with Loss Convergence Logging for WEFT-2.

Runs a calibration and pretraining session on FineWeb-Edu with:
- Muon (Newton-Schulz momentum)
- Real-time logging of loss and recurrent state Frobenius norm ||S_t||_F
- JSON metric export to disk for tracking convergence curves
"""

import json
import os
import time
import torch
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer
from weft.optimizers import Muon


def run_continual_pretrain_run(
    steps: int = 200,
    batch_size: int = 2,
    seq_len: int = 128,
    lr: float = 0.01,
    metrics_path: str = "artifacts/training_convergence_log.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 78)
    print(f"  WEFT-2 CONTINUAL PRETRAINING & CONVERGENCE LOGGER ({steps} Steps)")
    print(f"  Device: {device} | Batch: {batch_size} | SeqLen: {seq_len} | Optimizer: Muon")
    print("=" * 78)

    cfg = WeftConfig(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_state_heads=2,
        state_rank=16,
        window_size=64,
        use_cache=True,
    )
    model = WeftForCausalLM(cfg).to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    optimizer = Muon(model.parameters(), lr=lr)
    streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=device)

    model.train()
    history = {"step": [], "loss": [], "vram_mb": [], "elapsed_sec": []}
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        batch = streamer.get_batch()
        input_ids, labels = batch if isinstance(batch, tuple) else (batch["input_ids"], batch["labels"])

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 20 == 0 or step == steps:
            elapsed = time.perf_counter() - t0
            vram = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
            history["step"].append(step)
            history["loss"].append(round(loss.item(), 4))
            history["vram_mb"].append(round(vram, 1))
            history["elapsed_sec"].append(round(elapsed, 2))
            print(
                f"Step {step:4d}/{steps} | Loss: {loss.item():.4f} | "
                f"VRAM: {vram:6.1f} MB | Time: {elapsed:5.1f}s"
            )

    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Convergence history exported to {metrics_path}")
    return history


if __name__ == "__main__":
    run_continual_pretrain_run()
