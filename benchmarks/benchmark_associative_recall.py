"""
Phase 2 Benchmark: Associative Recall ("Needle in a Haystack")
Testing whether the Structured Block + Low-Rank Attention mechanism can
actually learn to retrieve a key-value pair placed far back in context.

Task:
  - Sequence contains random token pairs: Key_i -> Value_i.
  - At position k (distant past, e.g. position 100), a target pair is placed: (TARGET_KEY, SECRET_VALUE).
  - At the very end of the sequence (e.g. position 4096 or 8192), the query TARGET_KEY is presented.
  - The model must output SECRET_VALUE.

Comparison:
  1. Standard Transformer Layer (Softmax Multihead Attention)
  2. Structured Attention Layer (Window 256 + Rank 32 Global Bottleneck)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardAttentionModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, seq_len = input_ids.shape
        x = self.embed(input_ids)
        
        q = self.q_proj(x).view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        out = self.out_proj(attn_out)
        
        # Predict logits for the final query token
        return self.head(out[:, -1, :])


class StructuredAttentionModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, window_size: int = 256, rank: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.rank = rank

        self.proj_k = nn.Linear(d_model, rank, bias=False)
        self.proj_v = nn.Linear(d_model, rank, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, seq_len = input_ids.shape
        x = self.embed(input_ids)

        # 1. Local window
        pad_len = (self.window_size - (seq_len % self.window_size)) % self.window_size
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        total_len = x_pad.shape[1]
        n_windows = total_len // self.window_size

        x_win = x_pad.view(b * n_windows, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        local_out = F.scaled_dot_product_attention(x_win, x_win, x_win)
        local_out = local_out.transpose(1, 2).contiguous().view(b, total_len, self.d_model)
        if pad_len > 0:
            local_out = local_out[:, :seq_len, :]

        # 2. Global rank-k bottleneck
        k_comp = torch.bmm(self.proj_k(x).transpose(1, 2), x) / (seq_len ** 0.5)
        v_comp = self.proj_v(x)
        global_out = torch.bmm(v_comp, k_comp)

        out = self.out_proj(local_out + global_out)
        return self.head(out[:, -1, :])


def generate_batch(batch_size: int, seq_len: int, vocab_size: int, needle_pos: int, device: torch.device):
    """
    Generates a batch where:
      - Random filler tokens throughout.
      - At needle_pos: KEY token and VALUE token are placed.
      - At seq_len - 1: KEY token is presented.
      - Target is the corresponding VALUE token.
    """
    # Reserve tokens:
    # 0: pad, 1..50: keys, 51..100: values, 101+: filler
    keys = torch.randint(1, 50, (batch_size,), device=device)
    values = torch.randint(51, 100, (batch_size,), device=device)
    
    # Fill background with random distractor tokens
    tokens = torch.randint(101, vocab_size, (batch_size, seq_len), device=device)
    
    # Place needle at needle_pos
    tokens[:, needle_pos] = keys
    tokens[:, needle_pos + 1] = values
    
    # Query key at the end
    tokens[:, -1] = keys
    targets = values
    
    return tokens, targets


def train_and_eval(model: nn.Module, vocab_size: int, seq_len: int, needle_pos: int, device: torch.device, epochs: int = 100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    batch_size = 16
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(epochs):
        inputs, targets = generate_batch(batch_size, seq_len, vocab_size, needle_pos, device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total_evaluated = 0
    target_count = 200
    with torch.no_grad():
        while total_evaluated < target_count:
            curr_bs = min(batch_size, target_count - total_evaluated)
            inputs, targets = generate_batch(curr_bs, seq_len, vocab_size, needle_pos, device)
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_evaluated += curr_bs

    accuracy = (correct / total_evaluated) * 100
    return accuracy


def run_experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab_size = 500
    d_model = 256
    n_heads = 4

    print("=" * 80)
    print("PHASE 2: NEEDLE-IN-A-HAYSTACK ASSOCIATIVE RECALL TEST")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model dimensions: d_model={d_model}, heads={n_heads}, vocab={vocab_size}")
    print("=" * 80)
    print(f"{'Seq Length':>10} | {'Needle Distance':>16} | {'Standard SDPA Acc':>18} | {'Structured Acc (r=32)':>22}")
    print("-" * 80)

    # Test cases:
    # 1. Needle inside local window (distance = 100 < window_size 256)
    # 2. Needle far beyond local window (distance = 1000 >> window_size 256)
    # 3. Needle very far beyond local window (distance = 3500 >> window_size 256)
    test_cases = [
        (512, 100),    # Distance: 412 tokens
        (1024, 100),   # Distance: 924 tokens
        (2048, 100),   # Distance: 1948 tokens
        (4096, 100),   # Distance: 3996 tokens
    ]

    for seq_len, needle_pos in test_cases:
        distance = seq_len - needle_pos - 1

        # Train Standard SDPA
        torch.cuda.empty_cache()
        std_model = StandardAttentionModel(vocab_size, d_model, n_heads).to(device)
        std_acc = train_and_eval(std_model, vocab_size, seq_len, needle_pos, device, epochs=120)

        # Train Structured Attention
        torch.cuda.empty_cache()
        struct_model = StructuredAttentionModel(vocab_size, d_model, n_heads, window_size=256, rank=32).to(device)
        struct_acc = train_and_eval(struct_model, vocab_size, seq_len, needle_pos, device, epochs=120)

        print(f"{seq_len:>10} | {distance:>16} | {std_acc:>17.1f}% | {struct_acc:>21.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
