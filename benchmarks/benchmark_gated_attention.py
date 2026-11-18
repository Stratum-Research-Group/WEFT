"""
Phase 4: Adding Exponential Decay Gating to the Structured Attention Operator
Comparing:
  1. Standard Causal Transformer (Softmax Attention)
  2. Gated Causal Structured Attention (Window 64 + Gated Rank-32 State)

Mathematical Formulation:
  - Instead of raw cumulative sum: S_t = S_{t-1} + k_t^T x_t
  - We introduce a learnable decay factor per head/channel:
      gamma = sigmoid(decay_param)  (bounded in (0, 1))
      S_t = gamma * S_{t-1} + k_t^T x_t
  - This prevents old tokens from drowning out immediate syntactic context.
"""

import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load TinyShakespeare
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


class GatedStructuredAttention(nn.Module):
    """
    Causal Block-Window + Gated Exponential Decay Low-Rank Accumulator
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
        
        # Learnable decay gate per rank dimension
        # Initialized around logit(0.9) so memory lasts ~10-50 tokens by default
        self.decay_logits = nn.Parameter(torch.ones(rank) * 2.2)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        
        # 1. Causal local window attention
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

        # 2. Gated low-rank state: S_t = gamma * S_{t-1} + k_t^T x_t
        gamma = torch.sigmoid(self.decay_logits).view(1, 1, self.rank, 1)  # (1, 1, rank, 1)
        k_p = F.elu(self.proj_k(x)) + 1.0  # (b, s, rank)
        v_p = self.proj_v(x)               # (b, s, rank)
        
        # Step-wise recurrent scan across time dimension
        # (Sequential loop over s=256 is fast in PyTorch GPU execution)
        states = []
        state = torch.zeros(b, self.rank, d, device=x.device, dtype=x.dtype)
        for t in range(s):
            kt = k_p[:, t, :].unsqueeze(-1)  # (b, rank, 1)
            xt = x[:, t, :].unsqueeze(1)     # (b, 1, d)
            new_input = torch.bmm(kt, xt)    # (b, rank, d)
            gamma_t = gamma.squeeze(1)       # (1, rank, 1)
            state = gamma_t * state + new_input
            states.append(state)
            
        kv_states = torch.stack(states, dim=1)  # (b, s, rank, d)
        
        # Readout: v_p_t @ state_t
        global_out = torch.einsum("bsr,bsrd->bsd", v_p, kv_states) / (self.rank ** 0.5)

        return self.out(local_out + global_out)


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
            elif attn_type == "gated":
                attn = GatedStructuredAttention(d_model, n_heads, window_size=64, rank=32)
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


def train_and_eval(model_name: str, attn_type: str, steps: int = 1000):
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

    # 20 validation batches
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            xv, yv = get_batch("val", batch_size, block_size)
            _, vloss = model(xv, yv)
            val_losses.append(vloss.item())

    mean_val_loss = sum(val_losses) / len(val_losses)
    perplexity = math.exp(mean_val_loss)

    context = torch.tensor(encode("ROMEO:\n"), dtype=torch.long, device=device).unsqueeze(0)
    gen_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
    sample = decode(gen_tokens)

    return {
        "val_loss": mean_val_loss,
        "perplexity": perplexity,
        "total_time": total_time,
        "ms_per_step": (total_time / steps) * 1000,
        "peak_vram": peak_vram,
        "sample": sample,
    }


def main():
    print("=" * 80)
    print("PHASE 4: GATED EXPONENTIAL DECAY BENCHMARK (TINYSHAKESPEARE)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    res_gated = train_and_eval("Gated Structured Attention (Window 64 + Gated Rank 32)", "gated", steps=1000)

    print("\n" + "=" * 80)
    print("GATED STRUCTURED ATTENTION EVALUATION")
    print("=" * 80)
    print(f"Validation Loss:         {res_gated['val_loss']:.4f}")
    print(f"Validation Perplexity:   {res_gated['perplexity']:.2f}")
    print(f"Training Time (1000 st): {res_gated['total_time']:.2f} s")
    print(f"Peak VRAM:               {res_gated['peak_vram']:.1f} MB")
    print("=" * 80)

    print("\n[Sample Generation from Gated Structured Model]")
    print("-" * 50)
    print(res_gated["sample"])
    print("-" * 50)


if __name__ == "__main__":
    main()
