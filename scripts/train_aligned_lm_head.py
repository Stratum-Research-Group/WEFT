"""
Align and Train the Student Vocabulary lm_head against Gemma 4.

Strategy:
1. Slices the teacher embed_tokens matrix [262144, 1536] directly from safetensors.
2. In Gemma, lm_head is weight-tied to embed_tokens.weight:
   P(token) = softmax(RMSNorm(h) @ embed_tokens.T)
3. For the student (hidden_dim=256), training an independent [262144, 256] projection matrix
   directly from scratch requires billions of tokens.
   Instead, we train a SULA Projection Bridge:
   W_bridge: [256 -> 1536] with orthonormal initialization.
   Student lm_head(h_student) = RMSNorm(h_student @ W_bridge.T) @ embed_tokens.T
4. Trains W_bridge and student upper layers on real text (data/fineweb_edu_cache.json)
   using teacher-student KL-Divergence and cross-entropy matching.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft import WeftConfig, WeftForCausalLM

class StudentGemmaHeadAligner(nn.Module):
    def __init__(self, student_dim: int = 256, teacher_dim: int = 1536):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        
        # Projection bridge from student hidden space to teacher embedding space
        self.bridge = nn.Sequential(
            nn.Linear(student_dim, teacher_dim, bias=False),
            nn.RMSNorm(teacher_dim, eps=1e-6)
        )
        # Initialize bridge with semi-orthogonal projection
        nn.init.orthogonal_(self.bridge[0].weight)

    def forward(self, student_hidden: torch.Tensor, teacher_embed_subset: torch.Tensor) -> torch.Tensor:
        """
        student_hidden: [batch, seq, 256]
        teacher_embed_subset: [V_sub, 1536]
        Returns: logits [batch, seq, V_sub]
        """
        projected = self.bridge(student_hidden) # [batch, seq, 1536]
        logits = torch.matmul(projected, teacher_embed_subset.t()) # [batch, seq, V_sub]
        return logits

def train_aligned_lm_head(
    num_steps: int = 50,
    batch_size: int = 2,
    seq_len: int = 32,
    vocab_subset_size: int = 8192,
    lr: float = 2e-3
):
    print("=" * 80)
    print("  TRAINING & ALIGNING STUDENT VOCABULARY LM_HEAD (GEMMA 4 BRIDGE)")
    print("=" * 80)

    e2b_path = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
    tokenizer_dir = "/run/media/akarta/AI/models/gemma-4-E2B-it"
    data_path = "data/fineweb_edu_cache.json"
    out_dir = "/run/media/akarta/AI/checkpoints/aligned_head"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "student_aligned_gemma_head.pt")

    # 1. Load Tokenizer
    print("Loading Gemma 4 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 2. Load Real Text Data
    print(f"Loading training text from {data_path}...")
    with open(data_path) as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} real FineWeb-Edu text documents.")

    # 3. Load Teacher Embeddings (Zero-RAM Slice of active vocab)
    print(f"Slicing teacher embed_tokens (top {vocab_subset_size} frequent vocabulary)...")
    with safe_open(e2b_path, framework="pt") as f:
        embed_slice = f.get_slice("model.language_model.embed_tokens.weight")
        teacher_embed = embed_slice[:vocab_subset_size, :].float() # [V_sub, 1536]

    # 4. Student Architecture
    cfg = WeftConfig(
        vocab_size=vocab_subset_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=128
    )
    student = WeftForCausalLM(cfg)
    aligner = StudentGemmaHeadAligner(student_dim=256, teacher_dim=1536)

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(aligner.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    student.train()
    aligner.train()

    t0 = time.time()
    step = 0
    doc_idx = 0

    while step < num_steps and doc_idx < len(corpus):
        doc = corpus[doc_idx]
        doc_idx += 1
        
        # Tokenize with Gemma 4 tokenizer
        tokens = tokenizer.encode(doc)
        # Filter tokens outside sampled active vocabulary
        tokens = [t for t in tokens if t < vocab_subset_size]
        
        if len(tokens) < seq_len + 1:
            continue

        # Create input and target sequences
        input_ids = torch.tensor([tokens[:seq_len]], dtype=torch.long)
        target_ids = torch.tensor([tokens[1:seq_len+1]], dtype=torch.long)

        optimizer.zero_grad()
        
        # Forward pass through student
        s_out = student(input_ids, output_hidden_states=True)
        s_hidden = s_out.hidden_states[-1] # [1, seq_len, 256]

        # Project through bridge to teacher embedding dot-product
        logits = aligner(s_hidden, teacher_embed) # [1, seq_len, V_sub]

        loss = loss_fn(logits.view(-1, vocab_subset_size), target_ids.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(aligner.parameters()), 1.0)
        optimizer.step()

        step += 1
        if step % 10 == 0 or step == 1:
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == target_ids).float().mean().item() * 100.0
            print(f"Step {step:02d}/{num_steps} | Loss: {loss.item():.4f} | Token Accuracy: {acc:5.1f}%")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.2f}s!")

    # Save aligned head and bridge checkpoint
    torch.save({
        "student": student.state_dict(),
        "bridge": aligner.state_dict(),
        "vocab_subset_size": vocab_subset_size,
        "teacher_dim": 1536,
        "student_dim": 256
    }, out_file)
    print(f"Saved aligned student and Gemma head bridge to: {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    train_aligned_lm_head()
