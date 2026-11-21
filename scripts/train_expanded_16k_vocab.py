"""
Expanded 16,384-Token Vocabulary Alignment & Distillation Training:
Scales the active student vocabulary from 4,096 -> 16,384 tokens directly from
Gemma 4 E2B embeddings with zero-RAM slicing.
All checkpoints saved to NVMe /run/media/akarta/AI/checkpoints/aligned_head/
"""
import os
import json
import time
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer
from weft import WeftConfig, WeftForCausalLM

class ExpandedStudentAlignedHead(nn.Module):
    def __init__(self, student_hidden_dim: int = 256, teacher_dim: int = 1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_hidden_dim, 512, bias=False),
            nn.RMSNorm(512),
            nn.GELU(),
            nn.Linear(512, teacher_dim, bias=False),
            nn.RMSNorm(teacher_dim),
        )

    def forward(self, student_hidden: torch.Tensor, teacher_embed_subset: torch.Tensor) -> torch.Tensor:
        projected = self.proj(student_hidden) # [batch, seq, 1536]
        logits = torch.matmul(projected, teacher_embed_subset.t()) # [batch, seq, 16384]
        return logits

def train_16k_aligned_vocab():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 16384
    e2b_path = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
    tokenizer_dir = "/run/media/akarta/AI/models/gemma-4-E2B-it"
    out_dir = "/run/media/akarta/AI/checkpoints/aligned_head"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "student_aligned_gemma_16k_head.pt")

    print("=" * 80)
    print(f"  SCALING ACTIVE VOCABULARY TO {vocab_size} TOKENS ON CUDA ({device.upper()})")
    print("=" * 80)

    # 1. Zero-RAM Slice of 16,384 active tokens
    print(f"Slicing 16,384 teacher embeddings from {e2b_path}...")
    with safe_open(e2b_path, framework="pt") as f:
        slice_op = f.get_slice("model.language_model.embed_tokens.weight")
        teacher_embed = slice_op[:vocab_size, :].float().to(device)
    print(f"Teacher active embeddings loaded on {device}: {teacher_embed.shape} ({teacher_embed.nbytes / 1e6:.2f} MB)")

    # 2. Tokenizer & Corpus
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    data_path = "/run/media/akarta/AI/data/fluency_code_instruct_corpus.json"
    with open(data_path) as f:
        corpus = json.load(f)

    # Encode token stream bounded to 16,384
    raw_tokens = []
    for item in corpus[:200]:
        text = item if isinstance(item, str) else item.get("text", "")
        toks = tokenizer.encode(text)
        for t in toks:
            if t < vocab_size:
                raw_tokens.append(t)
    print(f"Active 16k token stream prepared: {len(raw_tokens):,} tokens.")

    # 3. Model Architecture
    cfg = WeftConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=128,
    )
    student = WeftForCausalLM(cfg).to(device)
    bridge = ExpandedStudentAlignedHead(student_hidden_dim=256, teacher_dim=1536).to(device)

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(bridge.parameters()), lr=3e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    seq_len = 32
    batch_size = 8
    num_steps = 300

    best_acc = 0.0
    start_t = time.time()

    for step in range(1, num_steps + 1):
        idx = ((step - 1) * batch_size * seq_len) % (len(raw_tokens) - batch_size * (seq_len + 1))
        batch_tokens = [raw_tokens[idx + i * (seq_len + 1) : idx + (i + 1) * (seq_len + 1)] for i in range(batch_size)]
        
        batch_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        input_ids = batch_tensor[:, :-1]
        targets = batch_tensor[:, 1:]

        optimizer.zero_grad()
        out = student(input_ids, output_hidden_states=True)
        h = out.hidden_states[-1]
        logits = bridge(h, teacher_embed)

        loss = criterion(logits.view(-1, vocab_size), targets.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item() * 100.0
        if acc > best_acc:
            best_acc = acc

        if step % 50 == 0 or step == num_steps:
            top5 = torch.topk(logits, k=5, dim=-1).indices
            top5_acc = (top5 == targets.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
            print(f"Step {step:04d}/{num_steps} | Loss: {loss.item():.4f} | Top-1: {acc:5.1f}% | Top-5: {top5_acc:5.1f}% (Best: {best_acc:5.1f}%)")

    elapsed = time.time() - start_t
    print(f"\n16k Vocabulary Alignment Finished in {elapsed:.2f}s! Best Accuracy: {best_acc:.2f}%")

    # Save checkpoint
    torch.save({
        "student_state_dict": student.state_dict(),
        "bridge_state_dict": bridge.state_dict(),
        "vocab_size": vocab_size,
        "best_acc": best_acc,
    }, out_file)
    print(f"Saved 16k aligned checkpoint to: {out_file} ({os.path.getsize(out_file) / 1e6:.2f} MB)")

if __name__ == "__main__":
    train_16k_aligned_vocab()
