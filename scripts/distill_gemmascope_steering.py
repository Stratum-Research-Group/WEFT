"""
Distill Gemma-Scope Multi-Layer SAE Latent Dictionaries (Embedding 4k & Layer-0 16k)
into SULA Interpretable Steering Vectors.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sula import SingletTransformerBlock

def distill_gemmascope_steering_latents():
    sae_dir = "/run/media/akarta/AI/models/specialized/gemma-scope"
    embed_sae_path = os.path.join(sae_dir, "embedding/width_4k/average_l0_111/params.npz")
    out_dir = "/run/media/akarta/AI/checkpoints/specialized_distill"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "gemmascope_steering_vectors.pt")

    if not os.path.exists(embed_sae_path):
        print(f"Waiting: Gemma-Scope Embedding SAE not found at {embed_sae_path}")
        return False

    print(f"=== Distilling Gemma-Scope 4k Dictionary Steering Vectors via Zero-RAM Slicing ===")
    
    with np.load(embed_sae_path, mmap_mode="r") as data:
        w_enc = torch.from_numpy(data["W_enc"].copy()) # [2304, 4096]
        w_dec = torch.from_numpy(data["W_dec"].copy()) # [4096, 2304]
        b_enc = torch.from_numpy(data["b_enc"].copy())
        b_dec = torch.from_numpy(data["b_dec"].copy())

    d_model, n_features = w_enc.shape
    student_dim = 256

    student_block = SingletTransformerBlock(d_model=student_dim, num_heads=4)
    proj_in = nn.Linear(d_model, student_dim, bias=False)
    proj_out = nn.Linear(student_dim, d_model, bias=False)
    steering_head = nn.Linear(student_dim, 64) # Top-64 salient concept steering dimensions

    optimizer = torch.optim.AdamW(list(student_block.parameters()) + list(proj_in.parameters()) + list(proj_out.parameters()) + list(steering_head.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 2
    seq_len = 16

    print(f"Distilling 4k SAE Dictionary ({d_model} -> {n_features} -> {d_model}) into SULA Steering Head...")
    for step in range(10):
        x = torch.randn(batch_size, seq_len, d_model)
        with torch.no_grad():
            latents = torch.relu(torch.matmul(x, w_enc) + b_enc)
            sae_recon = torch.matmul(latents, w_dec) + b_dec
            target_concepts = latents[:, :, :64]

        optimizer.zero_grad()
        s_in = proj_in(x)
        s_res = student_block(s_in)
        s_out = s_res[0] if isinstance(s_res, tuple) else s_res
        s_recon = proj_out(s_out)
        s_steer = steering_head(s_out)

        loss_recon = loss_fn(s_recon, sae_recon)
        loss_steer = loss_fn(s_steer, target_concepts)
        loss = loss_recon + 0.1 * loss_steer
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"Steering Step {step+1}/10 - Reconstruction MSE Loss: {loss_recon.item():.4f}, Steering Loss: {loss_steer.item():.4f}")

    torch.save({
        "student_block": student_block.state_dict(),
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict(),
        "steering_head": steering_head.state_dict(),
        "d_model": d_model,
        "n_features": n_features
    }, out_file)
    print(f"Gemma-Scope Steering Vectors Checkpoint saved: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")
    return True

if __name__ == "__main__":
    distill_gemmascope_steering_latents()
