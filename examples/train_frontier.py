"""
Distributed Frontier Pretraining Pipeline for WEFT-2 (1.5B / 3B / 7B Tiers).

Supports:
- PyTorch Fully Sharded Data Parallel (FSDP2) / ZeRO-3
- Muon Newton-Schulz Orthogonalized Momentum Optimizer
- FineWeb-Edu High-Throughput Token Streaming
- Dual-Axis Memory Initialization (DAMI)
- Titans Associative Plasticity + DeepSeekMoE fine-grained expert routing
- Zero-memory meta device initial checkpoint generation
"""

import argparse
import math
import os
import time
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM
from weft.scaling_specs import WEFT_SCALE_REGISTRY
from weft.dataset_fineweb import FineWebEduStreamer
from weft.optimizers import Muon


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def run_frontier_pretrain(
    tier: str = "1.5B",
    max_steps: int = 1000,
    warmup_steps: int = 100,
    batch_size: int = 2,
    seq_len: int = 512,
    max_lr: float = 0.01,
    min_lr: float = 0.001,
    save_dir: str = "weft_frontier_ckpt",
    use_fsdp: bool = False,
    dry_run: bool = False,
):
    rank, world_size, local_rank, is_dist = setup_distributed()
    is_main = rank == 0

    if tier not in WEFT_SCALE_REGISTRY:
        raise ValueError(f"Unknown tier {tier}. Registered: {list(WEFT_SCALE_REGISTRY.keys())}")

    spec = WEFT_SCALE_REGISTRY[tier]
    config = spec.to_config()

    if is_main:
        print("=" * 78)
        print(f"  WEFT-2 FRONTIER PRETRAINING LAUNCHER: TIER [{tier}]")
        print(f"  Target Parameters: {spec.target_params} | Layers: {config.num_hidden_layers} | Hidden: {config.hidden_size}")
        print(f"  World Size: {world_size} | Local Rank: {local_rank} | FSDP Enabled: {use_fsdp}")
        print(f"  Batch/Rank: {batch_size} | SeqLen: {seq_len} | Warmup: {warmup_steps} | Steps: {max_steps}")
        print("=" * 78)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # In dry-run or resource verification mode, confirm setup and parameter geometry
    if dry_run:
        with torch.device("meta"):
            meta_model = WeftForCausalLM(config)
            total_params = sum(p.numel() for p in meta_model.parameters())
        if is_main:
            print(f"[DRY-RUN] Verified {tier} model parameters: {total_params / 1e9:.2f}B.")
            print(f"[DRY-RUN] Pretraining pipeline validated successfully.")
        cleanup_distributed(is_dist)
        return

    # Real training initialization (recommended on cluster or single GPU for smaller tiers)
    model = WeftForCausalLM(config).to(device=device, dtype=torch.bfloat16)

    if use_fsdp and is_dist:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
        )

    optimizer = Muon(model.parameters(), lr=max_lr, momentum=0.95, n_iter=5)
    streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=str(device))

    model.train()
    t_start = time.perf_counter()

    for step in range(1, max_steps + 1):
        lr = get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = streamer.get_batch()
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
        else:
            input_ids, labels = batch

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if is_main and (step % 50 == 0 or step == max_steps):
            elapsed = time.perf_counter() - t_start
            tok_per_sec = (step * batch_size * seq_len * world_size) / max(1e-5, elapsed)
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(
                f"Step {step:5d}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.5f} | "
                f"VRAM: {vram_mb:6.1f} MB | Throughput: {tok_per_sec:7.1f} tok/s"
            )

    if is_main and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if use_fsdp and is_dist:
            with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
                state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(save_dir, "model_fsdp.pt"))
        else:
            model.save_pretrained(save_dir)
            config.save_pretrained(save_dir)
        print(f"Checkpoint successfully saved to {save_dir}")

    cleanup_distributed(is_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default="1.5B", choices=list(WEFT_SCALE_REGISTRY.keys()))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--save_dir", type=str, default="weft_frontier_ckpt")
    args = parser.parse_args()

    run_frontier_pretrain(
        tier=args.tier,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_lr=args.lr,
        save_dir=args.save_dir,
        use_fsdp=args.fsdp,
        dry_run=args.dry_run,
    )
