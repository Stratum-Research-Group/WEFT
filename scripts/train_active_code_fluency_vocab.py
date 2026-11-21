"""
Large-Scale Dynamic Vocabulary Code & Natural Language Distillation:
1. Gathers all unique tokens appearing in the Code & Instruction corpus (e.g. 8,192 active tokens).
2. Maps them to an active vocabulary tensor directly indexed from Gemma 4 embed_tokens.
3. Trains the student model and projection bridge to output natural syntax, keywords, and code tokens.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft import WeftConfig, WeftForCausalLM

class StudentGemmaBridge(nn.Module):
    def __init__(self, student_dim: int = 256, teacher_dim: int = 1536):
        super().__init__()
        self.fc1 = nn.Linear(student_dim, 512, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(512, teacher_dim, bias=False)
        self.norm = nn.RMSNorm(teacher_dim, eps=1e-6)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, h: torch.Tensor, embed_w: torch.Tensor) -> torch.Tensor:
        projected = self.norm(self.fc2(self.act(self.fc1(h))))
        logits = torch.matmul(projected, embed_w.t())
        return logits

def train_active_fluency():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    seq_len = 32
    max_steps = 800
    lr = 2e-3

    e2b_path = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
    tokenizer_dir = "/run/media/akarta/AI/models/gemma-4-E2B-it"
    data_path = "/run/media/akarta/AI/data/fluency_code_instruct_corpus.json"
    out_dir = "/run/media/akarta/AI/checkpoints/aligned_head"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "student_aligned_gemma_head.pt")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    with open(data_path) as f:
        corpus = json.load(f)

    # 1. Collect all distinct tokens across the corpus
    print("Collecting active vocabulary across code and technical instruction datasets...")
    raw_tokenized_docs = []
    vocab_counter = {}
    for doc in corpus:
        toks = tokenizer.encode(doc)
        raw_tokenized_docs.append(toks)
        for t in toks:
            vocab_counter[t] = vocab_counter.get(t, 0) + 1

    # Keep top 4,096 most frequent active tokens in the code & instruction corpus
    sorted_active_tokens = sorted(vocab_counter.keys(), key=lambda t: vocab_counter[t], reverse=True)[:4096]
    gemma_to_active = {g_id: active_idx for active_idx, g_id in enumerate(sorted_active_tokens)}
    active_vocab_size = len(sorted_active_tokens)
    print(f"Active domain vocabulary curated: {active_vocab_size:,} distinct tokens (includes Python/Rust/C++ keywords).")

    # 2. Slice teacher embeddings for these active tokens
    print("Slicing teacher embeddings for the active vocabulary directly into GPU memory...")
    active_tokens_tensor = torch.tensor(sorted_active_tokens, dtype=torch.long)
    with safe_open(e2b_path, framework="pt") as f:
        all_embed = f.get_tensor("model.language_model.embed_tokens.weight")
        teacher_embed = all_embed[active_tokens_tensor, :].float().to(device)

    # 3. Initialize Student & Bridge
    cfg = WeftConfig(
        vocab_size=active_vocab_size,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=128,
    )
    student = WeftForCausalLM(cfg).to(device)
    bridge = StudentGemmaBridge(student_dim=256, teacher_dim=1536).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(bridge.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Pre-tokenize dataset into active token IDs
    active_token_stream = []
    for doc_toks in raw_tokenized_docs:
        active_token_stream.extend([gemma_to_active[t] for t in doc_toks if t in gemma_to_active])

    tokens_tensor = torch.tensor(active_token_stream, dtype=torch.long, device=device)
    total_tokens = len(active_token_stream)
    print(f"Token stream prepared: {total_tokens:,} tokens mapped to active vocabulary.")

    best_acc = 0.0
    t0 = time.time()
    for step in range(1, max_steps + 1):
        start_indices = torch.randint(0, total_tokens - seq_len - 1, (batch_size,))
        batch_inputs = torch.stack([tokens_tensor[i : i + seq_len] for i in start_indices])
        batch_targets = torch.stack([tokens_tensor[i + 1 : i + seq_len + 1] for i in start_indices])

        optimizer.zero_grad()
        out = student(batch_inputs, output_hidden_states=True)
        h = out.hidden_states[-1]
        logits = bridge(h, teacher_embed)

        loss = loss_fn(logits.view(-1, active_vocab_size), batch_targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(bridge.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == batch_targets).float().mean().item() * 100.0
        top5_correct = torch.topk(logits, k=5, dim=-1).indices == batch_targets.unsqueeze(-1)
        top5_acc = top5_correct.any(dim=-1).float().mean().item() * 100.0

        if acc > best_acc:
            best_acc = acc

        if step % 50 == 0 or step == 1:
            print(f"Step {step:04d}/{max_steps} | Loss: {loss.item():.4f} | Top-1 Acc: {acc:5.1f}% | Top-5 Acc: {top5_acc:5.1f}% (Best: {best_acc:5.1f}%)")

    torch.save({
        "student": student.state_dict(),
        "bridge": bridge.state_dict(),
        "best_accuracy": best_acc,
        "vocab_size": active_vocab_size,
        "active_tokens": sorted_active_tokens,
        "steps_trained": step,
    }, out_file)
    print(f"\nTraining completed in {time.time()-t0:.2f}s! Best Accuracy: {best_acc:.2f}%")
    print(f"Saved active code-fluency checkpoint to: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    train_active_fluency()
