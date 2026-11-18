"""
Phase 3: Autoregressive Language Modeling Benchmark on Real Text (TinyShakespeare)
Comparing:
  1. Standard Causal Transformer (Softmax Multi-Head Attention)
  2. Causal Structured Attention (Causal Window + Online Low-Rank Causal Accumulator)

Metrics:
  - Training Loss & Validation Loss over 1,000 steps
  - Perplexity (exp(val_loss))
  - Wall-clock step latency (ms/step)
  - VRAM consumption (MB)
  - Sample generated text from both models
"""

import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and tokenize TinyShakespeare (character-level for fast, self-contained training)
with open("/home/akarta/Projects/0.WiP/Weft/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str, batch_size: int, block_size: int):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


# --- Model 1: Standard Causal Transformer Layer ---
class StandardCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out(out)


# --- Model 2: Causal Structured Attention Layer ---
class CausalStructuredAttention(nn.Module):
    """
    Causal Block-Window + Causal Low-Rank Cumulative Accumulator
    """
    def __init__(self, d_model: int, n_heads: int, window_size: int = 64, rank: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.rank = rank

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        
        self.proj_k = nn.Linear(d_model, rank, bias=False)
        self.proj_v = nn.Linear(d_model, rank, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        
        # 1. Causal sliding-block attention
        pad_len = (self.window_size - (s % self.window_size)) % self.window_size
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        tot_s = x_pad.shape[1]
        n_win = tot_s // self.window_size
        
        q_win = self.q(x_pad).view(b * n_win, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        k_win = self.k(x_pad).view(b * n_win, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        v_win = self.v(x_pad).view(b * n_win, self.window_size, self.n_heads, self.head_dim).transpose(1, 2)
        
        local_out = F.scaled_dot_product_attention(q_win, k_win, v_win, is_causal=True)
        local_out = local_out.transpose(1, 2).contiguous().view(b, tot_s, d)
        if pad_len > 0:
            local_out = local_out[:, :s, :]

        # 2. Causal online low-rank accumulation
        # k_p: (b, s, rank), v_p: (b, s, rank)
        k_p = F.elu(self.proj_k(x)) + 1.0  # Non-negative kernel feature map
        v_p = self.proj_v(x)
        
        # Outer-product state per step: outer(k_p_t, x_t) -> (b, s, rank, d)
        # Using cumulative sum along sequence dimension enforces strict causality!
        kv_step = torch.einsum("bsr,bsd->bsrd", k_p, x)
        kv_state = torch.cumsum(kv_step, dim=1)  # (b, s, rank, d)
        
        # Readout: v_p_t @ state_t -> (b, s, d)
        norm = torch.cumsum(k_p, dim=1)  # (b, s, rank)
        norm = torch.clamp(torch.einsum("bsr,bsr->bs", v_p, norm), min=1e-5).unsqueeze(-1)
        global_out = torch.einsum("bsr,bsrd->bsd", v_p, kv_state) / (norm ** 0.5)

        return self.out(local_out + global_out)


# --- Full Transformer Backbone ---
class TransformerBlock(nn.Module):
    def __init__(self, attn_module, d_model: int):
        super().__init__()
        self.attn = attn_module
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, attn_type: str, d_model: int = 256, n_heads: int = 4, n_layers: int = 4, block_size: int = 256):
        super().__init__()
        self.block_size = block_size
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)

        blocks = []
        for _ in range(n_layers):
            if attn_type == "standard":
                attn = StandardCausalAttention(d_model, n_heads)
            elif attn_type == "structured":
                attn = CausalStructuredAttention(d_model, n_heads, window_size=64, rank=32)
            blocks.append(TransformerBlock(attn, d_model))

        self.blocks = nn.Sequential(*blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, s = idx.shape
        pos = torch.arange(0, s, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def train_model(model_name: str, attn_type: str, steps: int = 1000):
    print(f"\nTraining {model_name}...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = LanguageModel(attn_type=attn_type, d_model=256, n_heads=4, n_layers=4, block_size=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    batch_size = 32
    block_size = 256

    t0 = time.perf_counter()
    for step in range(steps):
        model.train()
        xb, yb = get_batch("train", batch_size, block_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch("val", batch_size, block_size)
                _, vloss = model(xv, yv)
            print(f"  Step {step:>4}: Train Loss = {loss.item():.4f} | Val Loss = {vloss.item():.4f}")

    t1 = time.perf_counter()
    total_time = t1 - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Final evaluation on 10 validation batches
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            xv, yv = get_batch("val", batch_size, block_size)
            _, vloss = model(xv, yv)
            val_losses.append(vloss.item())

    mean_val_loss = sum(val_losses) / len(val_losses)
    perplexity = math.exp(mean_val_loss)

    # Sample generation
    context = torch.tensor(encode("ROMEO:\n"), dtype=torch.long, device=device).unsqueeze(0)
    gen_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
    generated_sample = decode(gen_tokens)

    return {
        "val_loss": mean_val_loss,
        "perplexity": perplexity,
        "total_time": total_time,
        "ms_per_step": (total_time / steps) * 1000,
        "peak_vram": peak_vram,
        "sample": generated_sample,
    }


def main():
    print("=" * 80)
    print("PHASE 3: NATURAL LANGUAGE MODELING BENCHMARK (TINYSHAKESPEARE)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    res_std = train_model("Standard Causal Transformer (Softmax Attention)", "standard", steps=1000)
    res_struct = train_model("Causal Structured Attention (Window 64 + Rank 32)", "structured", steps=1000)

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD LANGUAGE MODELING RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Standard Transformer':<22} | {'Structured Attention':<22}")
    print("-" * 80)
    print(f"{'Validation Loss':<25} | {res_std['val_loss']:<22.4f} | {res_struct['val_loss']:<22.4f}")
    print(f"{'Validation Perplexity':<25} | {res_std['perplexity']:<22.2f} | {res_struct['perplexity']:<22.2f}")
    print(f"{'Training Time (1000 steps)':<25} | {res_std['total_time']:<22.2f} s | {res_struct['total_time']:<22.2f} s")
    print(f"{'Step Latency':<25} | {res_std['ms_per_step']:<22.2f} ms | {res_struct['ms_per_step']:<22.2f} ms")
    print(f"{'Peak VRAM':<25} | {res_std['peak_vram']:<22.1f} MB | {res_struct['peak_vram']:<22.1f} MB")
    print("=" * 80)

    print("\n[Sample Generation: Standard Transformer]")
    print("-" * 50)
    print(res_std["sample"])
    print("-" * 50)

    print("\n[Sample Generation: Structured Attention]")
    print("-" * 50)
    print(res_struct["sample"])
    print("-" * 50)


if __name__ == "__main__":
    main()
