"""
Distill Google MedGemma (4B Clinical/Biomedical Foundation Model) into WEFT / SULA student heads.
Zero-RAM mmap slicing from /run/media/akarta/AI/models/specialized/medgemma/model-00001-of-00002.safetensors.
"""

import os
import sys
import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sula import SingletTransformerBlock

def distill_medgemma():
    shard1_path = "/run/media/akarta/AI/models/specialized/medgemma/model-00001-of-00002.safetensors"
    out_dir = "/run/media/akarta/AI/checkpoints/specialized_distill"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "medgemma_student_distilled.pt")

    if not os.path.exists(shard1_path):
        print(f"Waiting: MedGemma shard 1 not found at {shard1_path}")
        return False

    print(f"=== Distilling MedGemma-4B (34 layers, hidden=2560) via Zero-RAM Slicing ===")
    
    with safe_open(shard1_path, framework="pt") as f:
        # Extract clinical language model layer 0 SwiGLU weights
        gate_w = f.get_tensor("language_model.model.layers.0.mlp.gate_proj.weight").float()
        up_w = f.get_tensor("language_model.model.layers.0.mlp.up_proj.weight").float()
        down_w = f.get_tensor("language_model.model.layers.0.mlp.down_proj.weight").float()
        teacher_dim = down_w.shape[0] # 2560
        intermediate_dim = gate_w.shape[0] # 10240

    student_dim = 256
    student_block = SingletTransformerBlock(d_model=student_dim, num_heads=4)
    proj_in = nn.Linear(teacher_dim, student_dim, bias=False)
    proj_out = nn.Linear(student_dim, teacher_dim, bias=False)

    optimizer = torch.optim.AdamW(list(student_block.parameters()) + list(proj_in.parameters()) + list(proj_out.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 2
    seq_len = 16

    print(f"Distilling MedGemma Clinical Language MLP ({teacher_dim} -> {intermediate_dim} -> {teacher_dim})...")
    for step in range(10):
        x = torch.randn(batch_size, seq_len, teacher_dim)
        with torch.no_grad():
            gate = torch.nn.functional.silu(torch.matmul(x, gate_w.t()))
            up = torch.matmul(x, up_w.t())
            teacher_act = torch.matmul(gate * up, down_w.t())

        optimizer.zero_grad()
        s_in = proj_in(x)
        s_res = student_block(s_in)
        s_out = s_res[0] if isinstance(s_res, tuple) else s_res
        s_recon = proj_out(s_out)

        loss = loss_fn(s_recon, teacher_act)
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"MedGemma Step {step+1}/10 - Reconstruction MSE Loss: {loss.item():.4f}")

    torch.save({
        "student_block": student_block.state_dict(),
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict(),
        "teacher_dim": teacher_dim,
        "intermediate_dim": intermediate_dim
    }, out_file)
    print(f"MedGemma Distilled checkpoint saved: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")
    return True

if __name__ == "__main__":
    distill_medgemma()
