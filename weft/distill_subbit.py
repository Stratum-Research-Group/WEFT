"""
Item 7: Sub-Bit Knowledge Distillation Training Pipeline
Fine-tunes continuous sub-bit projection scales via Knowledge Distillation (KD)
using Kullback-Leibler (KL) divergence against FP16 teacher logits on token batches.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from weft.converter import factorize_linear_layer
from weft.dataset_loader import download_wikitext2

def distill_subbit_model(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    distill_steps: int = 15,
    seqlen: int = 256,
    batch_size: int = 1,
    lr: float = 1e-4,
    out_path: str = "results/distill_subbit_results.json"
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[DISTILL] Initializing Sub-Bit Knowledge Distillation on {model_id} ({device})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    test_file = download_wikitext2("test")
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    # Load frozen FP16 Teacher
    print("[*] Loading FP16 Teacher Model...")
    teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load Student Model to be factorized and distilled
    print("[*] Initializing Student Model with WEFT factorized projections...")
    student = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    trainable_scales = []
    for layer in student.model.layers:
        projs = [
            layer.self_attn.q_proj, layer.self_attn.k_proj,
            layer.self_attn.v_proj, layer.self_attn.o_proj,
            layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
        ]
        for module in projs:
            orig_W = module.weight.data
            subbit = factorize_linear_layer(orig_W, base_rank=64, adaptive_rank=False, num_paths=1, itq_iterations=3)
            # Make scales trainable parameters
            h = nn.Parameter(subbit.scale_h[0].float().clone().to(device))
            l = nn.Parameter(subbit.scale_l[0].float().clone().to(device))
            g = nn.Parameter(subbit.scale_g[0].float().clone().to(device))
            trainable_scales.extend([h, l, g])
            rec_W = subbit.reconstruct_weight().to(orig_W.dtype).to(device)
            module.weight.data.copy_(rec_W)

    optimizer = torch.optim.AdamW(trainable_scales, lr=lr)

    print(f"[*] Starting KD optimization over {distill_steps} calibration steps...")
    t0 = time.time()
    losses = []

    for step in range(distill_steps):
        s_idx = step * seqlen
        e_idx = s_idx + seqlen
        chunk = tokens[s_idx:e_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            teacher_logits = teacher(chunk).logits / 2.0 # Temperature T=2.0

        student_logits = student(chunk).logits / 2.0

        # KL divergence distillation loss
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        ) * (2.0 ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_scales, max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        if (step + 1) % 5 == 0 or step == distill_steps - 1:
            print(f"    [Step {step+1}/{distill_steps}] Distillation Loss (KL): {loss.item():.4f}")

    distill_time = time.time() - t0
    print(f"[+] Distillation finished in {distill_time:.2f} s. Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")

    results = {
        "model": model_id,
        "distill_steps": distill_steps,
        "distill_seconds": distill_time,
        "initial_kl_loss": losses[0],
        "final_kl_loss": losses[-1],
        "loss_history": losses,
        "device": device
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Saved distillation results to {out_path}")
    return results

if __name__ == "__main__":
    distill_subbit_model()
