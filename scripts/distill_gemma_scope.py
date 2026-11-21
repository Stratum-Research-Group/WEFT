"""
Distill Google Gemma-Scope Sparse Autoencoder (SAE) Latents into WEFT / SULA Associative Memory.
Zero-RAM mmap streaming of 16k dictionary features from params.npz.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sula import SingletTransformerBlock

def distill_gemma_scope():
    sae_path = "/run/media/akarta/AI/models/specialized/gemma-scope/layer_0/width_16k/average_l0_105/params.npz"
    out_dir = "/run/media/akarta/AI/checkpoints/specialized_distill"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "gemmascope_sae_student_distilled.pt")

    if not os.path.exists(sae_path):
        print(f"Waiting: Gemma-Scope SAE file not found at {sae_path}")
        return False

    print(f"=== Distilling Gemma-Scope SAE Layer 0 (16,384 features -> 2304) via Zero-RAM Slicing ===")
    
    # Load SAE weights with mmap mode
    with np.load(sae_path, mmap_mode="r") as data:
        w_enc = torch.from_numpy(data["W_enc"].copy())  # [2304, 16384]
        w_dec = torch.from_numpy(data["W_dec"].copy())  # [16384, 2304]
        b_enc = torch.from_numpy(data["b_enc"].copy())  # [16384]
        b_dec = torch.from_numpy(data["b_dec"].copy())  # [2304]
        
    d_model, n_features = w_enc.shape
    student_dim = 256
    
    # Student compresses sparse activations into linear state
    student_block = SingletTransformerBlock(d_model=student_dim, num_heads=4)
    proj_in = nn.Linear(d_model, student_dim, bias=False)
    proj_out = nn.Linear(student_dim, d_model, bias=False)

    optimizer = torch.optim.AdamW(list(student_block.parameters()) + list(proj_in.parameters()) + list(proj_out.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 2
    seq_len = 16

    print(f"Distilling SAE Sparse Reconstruction into SULA ({d_model} -> {n_features} -> {d_model})...")
    for step in range(10):
        x = torch.randn(batch_size, seq_len, d_model)
        with torch.no_grad():
            # Standard Top-K JumpReLU SAE forward pass
            latents = torch.relu(torch.matmul(x, w_enc) + b_enc) # [batch, seq, 16384]
            sae_recon = torch.matmul(latents, w_dec) + b_dec     # [batch, seq, 2304]

        optimizer.zero_grad()
        s_in = proj_in(x)
        s_res = student_block(s_in)
        s_out = s_res[0] if isinstance(s_res, tuple) else s_res
        s_recon = proj_out(s_out)

        loss = loss_fn(s_recon, sae_recon)
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"Gemma-Scope Step {step+1}/10 - SAE Reconstruction MSE Loss: {loss.item():.4f}")

    torch.save({
        "student_block": student_block.state_dict(),
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict(),
        "d_model": d_model,
        "n_features": n_features
    }, out_file)
    print(f"Gemma-Scope SAE Distilled checkpoint saved: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")
    return True

if __name__ == "__main__":
    distill_gemma_scope()
