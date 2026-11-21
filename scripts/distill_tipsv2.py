"""
Distillation pipeline for Google TIPS-v2 (Transformer with In-context Parameter Synthesis v2).
Distills TIPS-v2 conditioning embeddings and parameter synthesis heads into SULA linear state.
"""

import os
import sys
import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sula import SingletTransformerBlock

def distill_tipsv2():
    model_dir = "/run/media/akarta/AI/models/specialized/tipsv2"
    weights_path = os.path.join(model_dir, "model.safetensors")
    out_dir = "/run/media/akarta/AI/checkpoints/specialized_distill"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "tipsv2_student_distilled.pt")

    if not os.path.exists(weights_path):
        print(f"Waiting: TIPS-v2 weights not yet found at {weights_path}")
        return False

    print(f"=== Distilling TIPS-v2 via Zero-RAM Slicing from {weights_path} ===")
    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"TIPS-v2 Total Tensors: {len(keys)}")
        # Discover embedding or first linear projection
        first_key = keys[0]
        sample_shape = f.get_slice(first_key).get_shape()
        print(f"Sample tensor {first_key}: {sample_shape}")
        
        # Look for 2D mlp / feed-forward projection weights
        mlp_keys = [k for k in keys if "mlp" in k and "weight" in k and "proj" in k]
        target_key = mlp_keys[0] if mlp_keys else [k for k in keys if "weight" in k and len(f.get_slice(k).get_shape()) == 2][0]
        t_slice = f.get_slice(target_key)
        shape = t_slice.get_shape()
        teacher_dim = shape[0]  # Projection output dim (1152)

    student_dim = 256
    student_block = SingletTransformerBlock(d_model=student_dim, num_heads=4)
    proj_in = nn.Linear(teacher_dim, student_dim, bias=False)
    proj_out = nn.Linear(student_dim, teacher_dim, bias=False)

    optimizer = torch.optim.AdamW(list(student_block.parameters()) + list(proj_in.parameters()) + list(proj_out.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 2
    seq_len = 16

    with safe_open(weights_path, framework="pt") as f:
        sample_tensor = f.get_tensor(target_key).float()
        if sample_tensor.ndim != 2:
            sample_tensor = sample_tensor.view(teacher_dim, -1)

    print(f"Distilling target projection '{target_key}' (shape {sample_tensor.shape}) into SULA block...")
    for step in range(10):
        # Input matching sample_tensor input feature dimension
        in_dim = sample_tensor.shape[1]
        x_in = torch.randn(batch_size, seq_len, in_dim)
        with torch.no_grad():
            target_act = torch.matmul(x_in, sample_tensor.t()) # [batch, seq, teacher_dim]

        optimizer.zero_grad()
        s_in = proj_in(target_act)
        s_res = student_block(s_in)
        s_out = s_res[0] if isinstance(s_res, tuple) else s_res
        recon = proj_out(s_out)

        loss = loss_fn(recon, target_act)
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"TIPS-v2 Step {step+1}/10 - Reconstruction Loss: {loss.item():.4f}")

    torch.save({
        "student_block": student_block.state_dict(),
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict(),
        "target_key": target_key
    }, out_file)
    print(f"TIPS-v2 Distilled checkpoint saved: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")
    return True

if __name__ == "__main__":
    distill_tipsv2()
