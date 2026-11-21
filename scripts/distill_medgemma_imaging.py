"""
Distill MedGemma Clinical Medical Imaging Tower into WEFT / SULA Diagnostic Adapter.
Zero-RAM mmap slicing from /run/media/akarta/AI/models/specialized/medgemma/model-00001-of-00002.safetensors.
"""

import os
import sys
import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sula import SingletTransformerBlock

def distill_medgemma_imaging():
    weights_path = "/run/media/akarta/AI/models/specialized/medgemma/model-00001-of-00002.safetensors"
    out_dir = "/run/media/akarta/AI/checkpoints/specialized_distill"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "medgemma_imaging_student.pt")

    if not os.path.exists(weights_path):
        print(f"Waiting: MedGemma weights not found at {weights_path}")
        return False

    print(f"=== Distilling MedGemma Diagnostic Imaging Tower via Zero-RAM Slicing ===")
    
    with safe_open(weights_path, framework="pt") as f:
        # Extract clinical vision encoder layer 0 MLP
        w_fc1 = f.get_tensor("vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight").float() # [4304, 1152]
        w_fc2 = f.get_tensor("vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight").float() # [1152, 4304]
        b_fc1 = f.get_tensor("vision_tower.vision_model.encoder.layers.0.mlp.fc1.bias").float()
        b_fc2 = f.get_tensor("vision_tower.vision_model.encoder.layers.0.mlp.fc2.bias").float()
        teacher_dim = w_fc1.shape[1]      # 1152
        intermediate_dim = w_fc1.shape[0] # 4304

    student_dim = 256
    student_block = SingletTransformerBlock(d_model=student_dim, num_heads=4)
    proj_in = nn.Linear(teacher_dim, student_dim, bias=False)
    proj_out = nn.Linear(student_dim, teacher_dim, bias=False)

    optimizer = torch.optim.AdamW(list(student_block.parameters()) + list(proj_in.parameters()) + list(proj_out.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 2
    seq_len = 16 # 16 patch tokens

    print(f"Distilling Clinical Imaging MLP ({teacher_dim} -> {intermediate_dim} -> {teacher_dim})...")
    for step in range(10):
        # Simulated medical imaging patch embeddings (e.g. Chest X-Ray / CT patches)
        x = torch.randn(batch_size, seq_len, teacher_dim)
        with torch.no_grad():
            fc1_act = torch.nn.functional.gelu(torch.matmul(x, w_fc1.t()) + b_fc1)
            teacher_act = torch.matmul(fc1_act, w_fc2.t()) + b_fc2

        optimizer.zero_grad()
        s_in = proj_in(x)
        s_res = student_block(s_in)
        s_out = s_res[0] if isinstance(s_res, tuple) else s_res
        s_recon = proj_out(s_out)

        loss = loss_fn(s_recon, teacher_act)
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"MedGemma Imaging Step {step+1}/10 - Reconstruction MSE Loss: {loss.item():.4f}")

    torch.save({
        "student_block": student_block.state_dict(),
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict(),
        "teacher_dim": teacher_dim,
        "intermediate_dim": intermediate_dim
    }, out_file)
    print(f"MedGemma Clinical Imaging Checkpoint saved: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")
    return True

if __name__ == "__main__":
    distill_medgemma_imaging()
