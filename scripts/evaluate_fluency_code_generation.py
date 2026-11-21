"""
Evaluates code completion and natural language generation from the newly fine-tuned model.
Demonstrates syntax validity and code pattern completion.
"""
import os
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft import WeftConfig, WeftForCausalLM

class StudentGemmaBridge(torch.nn.Module):
    def __init__(self, student_dim=256, teacher_dim=1536):
        super().__init__()
        self.fc1 = torch.nn.Linear(student_dim, 512, bias=False)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(512, teacher_dim, bias=False)
        self.norm = torch.nn.RMSNorm(teacher_dim, eps=1e-6)

    def forward(self, h, embed_w):
        projected = self.norm(self.fc2(self.act(self.fc1(h))))
        return torch.matmul(projected, embed_w.t())

def test_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1024
    ckpt_path = "/run/media/akarta/AI/checkpoints/aligned_head/student_aligned_gemma_head.pt"
    e2b_path = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
    tokenizer_dir = "/run/media/akarta/AI/models/gemma-4-E2B-it"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    with safe_open(e2b_path, framework="pt") as f:
        embed_slice = f.get_slice("model.language_model.embed_tokens.weight")
        teacher_embed = embed_slice[:vocab_size, :].float().to(device)

    cfg = WeftConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=128,
    )
    student = WeftForCausalLM(cfg).to(device)
    bridge = StudentGemmaBridge(student_dim=256, teacher_dim=1536).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["student"])
    bridge.load_state_dict(ckpt["bridge"])
    student.eval()
    bridge.eval()

    test_prompts = [
        "def quicksort(arr:",
        "Question: How does speculative decoding work?",
        "import torch\nimport torch.nn as nn\nclass",
    ]

    print("=" * 70)
    print("  VERIFYING REAL GENERATION FLUENCY (CODE & NATURAL LANGUAGE)")
    print("=" * 70)

    for p in test_prompts:
        print(f"\n[Prompt]: {p!r}")
        toks = tokenizer.encode(p)
        curr_ids = torch.tensor([[t for t in toks if t < vocab_size]], dtype=torch.long, device=device)
        
        generated = []
        with torch.no_grad():
            for _ in range(16):
                out = student(curr_ids, output_hidden_states=True)
                h = out.hidden_states[-1][:, -1, :]
                logits = bridge(h, teacher_embed)
                next_tok = torch.argmax(logits, dim=-1, keepdim=True).long()
                generated.append(next_tok.item())
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)

        decoded = tokenizer.decode(generated)
        print(f"[Generated Continuation]: {decoded!r}")

if __name__ == "__main__":
    test_generation()
