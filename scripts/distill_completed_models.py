"""
Automated Pipeline: Distill Gemma 4 E4B strata and Gemma 4 26B-A4B Top-8 Experts
into WEFT / SULA zero-RAM checkpoints on /run/media/akarta/AI.
"""

import os
import sys
import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft import WeftConfig, WeftForCausalLM
from sula import SingletTransformerBlock

def distill_gemma4_e4b_strata():
    weights_path = "/run/media/akarta/AI/models/gemma-4-E4B-it/model.safetensors"
    out_dir = "/run/media/akarta/AI/checkpoints/gemma4_e4b_distill"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=== Distilling Gemma 4 E4B (42 layers, hidden=2560) via Zero-RAM Slicing ===")
    
    # We distill 5 landmark strata representing input, early-mid, mid, mid-deep, and final layers
    strata = [0, 10, 20, 30, 41]
    
    cfg = WeftConfig(
        vocab_size=262144,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=len(strata),
        num_attention_heads=8,
        max_position_embeddings=128
    )
    student = WeftForCausalLM(cfg)
    proj_in = nn.Linear(2560, 256, bias=False)
    proj_out = nn.Linear(256, 2560, bias=False)
    
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(proj_in.parameters()) + list(proj_out.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    batch_size = 2
    seq_len = 16
    
    for epoch in range(10):
        total_loss = 0.0
        for idx, layer_idx in enumerate(strata):
            with safe_open(weights_path, framework="pt") as f:
                gate_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight").float()
                up_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight").float()
                down_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight").float()
                
            x_in = torch.randn(batch_size, seq_len, 2560)
            gate = torch.nn.functional.silu(torch.matmul(x_in, gate_w.t()))
            up = torch.matmul(x_in, up_w.t())
            teacher_act = torch.matmul(gate * up, down_w.t())
            
            optimizer.zero_grad()
            student_x = proj_in(x_in)
            student_tokens = torch.randint(0, 1000, (batch_size, seq_len))
            s_out = student(student_tokens, output_hidden_states=True)
            s_hidden = s_out.hidden_states[min(idx + 1, len(s_out.hidden_states) - 1)]
            
            recon = proj_out(s_hidden)
            loss = loss_fn(recon, teacher_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"E4B Stratum Epoch {epoch+1}/10 - Mean MSE Loss: {total_loss / len(strata):.4f}")
            
    out_file = os.path.join(out_dir, "student_e4b_distilled.pt")
    torch.save({
        "student": student.state_dict(),
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict(),
        "strata": strata
    }, out_file)
    print(f"Gemma 4 E4B Distilled checkpoint saved: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")

def distill_gemma4_26b_top8_moe():
    shard1_path = "/run/media/akarta/AI/models/gemma-4-26B-A4B-it/model-00001-of-00002.safetensors"
    out_dir = "/run/media/akarta/AI/checkpoints/gemma4_moe_distill"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=== Distilling Gemma 4 26B-A4B Live Top-8 MoE Shard 1 via Zero-RAM Slicing ===")
    from scripts.distill_gemma4_moe_experts import Gemma4SparseMoEProber
    
    prober = Gemma4SparseMoEProber(shard1_path, top_k=8)
    student_block = SingletTransformerBlock(d_model=256, num_heads=4)
    proj_in = nn.Linear(2816, 256, bias=False)
    proj_out = nn.Linear(256, 2816, bias=False)
    
    optimizer = torch.optim.AdamW(list(student_block.parameters()) + list(proj_in.parameters()) + list(proj_out.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    batch_size = 2
    seq_len = 8
    
    for step in range(10):
        x = torch.randn(batch_size, seq_len, 2816)
        with torch.no_grad():
            teacher_moe_act = prober.probe_sparse_moe_layer(x, layer_idx=0)
            
        optimizer.zero_grad()
        s_in = proj_in(x)
        s_res = student_block(s_in)
        s_out = s_res[0] if isinstance(s_res, tuple) else s_res
        s_recon = proj_out(s_out)
        
        loss = loss_fn(s_recon, teacher_moe_act)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"MoE Step {step+1}/10 - Reconstruction Loss: {loss.item():.4f}")
            
    out_file = os.path.join(out_dir, "student_moe_top8_distilled.pt")
    torch.save({
        "student_block": student_block.state_dict(),
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict()
    }, out_file)
    print(f"Gemma 4 26B-A4B Top-8 MoE Distilled checkpoint saved: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    distill_gemma4_e4b_strata()
    distill_gemma4_26b_top8_moe()
