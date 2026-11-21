"""
Evaluates Code & Natural Language Generation with the 4,096 Active Vocabulary.
"""
import os
import sys
import torch
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
    ckpt_path = "/run/media/akarta/AI/checkpoints/aligned_head/student_aligned_gemma_head.pt"
    e2b_path = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
    tokenizer_dir = "/run/media/akarta/AI/models/gemma-4-E2B-it"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    active_tokens = ckpt["active_tokens"]
    active_vocab_size = len(active_tokens)
    gemma_to_active = {g: i for i, g in enumerate(active_tokens)}

    active_tokens_tensor = torch.tensor(active_tokens, dtype=torch.long)
    with safe_open(e2b_path, framework="pt") as f:
        all_embed = f.get_tensor("model.language_model.embed_tokens.weight")
        teacher_embed = all_embed[active_tokens_tensor, :].float().to(device)

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

    student.load_state_dict(ckpt["student"])
    bridge.load_state_dict(ckpt["bridge"])
    student.eval()
    bridge.eval()

    test_prompts = [
        "def quicksort(arr",
        "Question: How does speculative decoding work?",
        "import torch",
    ]

    print("=" * 70)
    print("  VERIFYING REAL GENERATION FLUENCY (CODE & NATURAL LANGUAGE)")
    print("=" * 70)

    for p in test_prompts:
        toks = tokenizer.encode(p)
        active_toks = [gemma_to_active[t] for t in toks if t in gemma_to_active]
        if not active_toks:
            active_toks = [0]
        curr_ids = torch.tensor([active_toks], dtype=torch.long, device=device)
        
        generated_gemma_tokens = []
        with torch.no_grad():
            for _ in range(12):
                out = student(curr_ids, output_hidden_states=True)
                h = out.hidden_states[-1][:, -1, :]
                logits = bridge(h, teacher_embed)
                next_active_idx = torch.argmax(logits, dim=-1, keepdim=True).long()
                next_gemma_id = active_tokens[next_active_idx.item()]
                generated_gemma_tokens.append(next_gemma_id)
                curr_ids = torch.cat([curr_ids, next_active_idx], dim=1)

        decoded = tokenizer.decode(generated_gemma_tokens)
        print(f"\n[Prompt]: {p!r}")
        print(f"[Generated Continuation]: {decoded!r}")

if __name__ == "__main__":
    test_generation()
