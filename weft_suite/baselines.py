"""
Track A Model Baselines:
Implements matched architectures for comparative evaluation:
  M1: WEFT base (local sliding-window + recurrent state)
  M2: WEFT local-only (global_path="off")
  M3: WEFT state-only (local_path="off")
  M4: Standard Transformer with full causal attention & RoPE
  M5: Transformer sliding-window only
All models built at matched parameter counts within +-5% tolerance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from weft.modeling_weft import WeftForCausalLM, WeftRMSNorm
from weft_suite.common import get_tiny_config

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1) # (1, 1, s, d)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class BaselineMLP(nn.Module):
    """SwiGLU feed-forward layer matching WeftMLP parameter structure."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class CausalSelfAttention(nn.Module):
    """Multi-head attention supporting full causal attention or causal sliding window."""
    def __init__(self, cfg, sliding_window=None):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.sliding_window is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            row = torch.arange(s, device=x.device).unsqueeze(1)
            col = torch.arange(s, device=x.device).unsqueeze(0)
            causal_mask = col > row
            window_mask = (row - col) >= self.sliding_window
            attn_mask = (causal_mask | window_mask).unsqueeze(0).unsqueeze(1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=~attn_mask)

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Decoder Block."""
    def __init__(self, cfg, sliding_window=None, intermediate_size=None):
        super().__init__()
        self.ln1 = WeftRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg, sliding_window=sliding_window)
        self.ln2 = WeftRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        inter = intermediate_size or cfg.intermediate_size
        self.mlp = BaselineMLP(cfg.hidden_size, inter)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BaselineTransformer(nn.Module):
    """M4 / M5 Standard Causal Transformer with parameter matching and RoPE."""
    def __init__(self, cfg, sliding_window=None, intermediate_size=139):
        super().__init__()
        self.cfg = cfg
        self.sliding_window = sliding_window
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, sliding_window=sliding_window, intermediate_size=intermediate_size)
            for _ in range(cfg.num_hidden_layers)
        ])
        self.norm = WeftRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def compute_matching_intermediate_size(target_params, cfg, num_attn_heads=None, is_baseline=True):
    """
    Analytically derives intermediate_size for SwiGLU MLP so that total used parameters
    match target_params exactly or within <= 1%.
    """
    hidden = cfg.hidden_size
    layers = cfg.num_hidden_layers
    vocab = cfg.vocab_size

    # Parameter components
    embed_and_norm = vocab * hidden + hidden # tied embed + lm_head + final norm
    if is_baseline:
        # ln1, ln2, q, k, v, out_proj per layer
        fixed_per_layer = 2 * hidden + 4 * (hidden ** 2)
    else:
        fixed_per_layer = 2 * hidden

    fixed_total = embed_and_norm + layers * fixed_per_layer
    mlp_factor_per_inter = layers * 3 * hidden
    computed_inter = max(16, round((target_params - fixed_total) / mlp_factor_per_inter))
    return computed_inter

def get_model_family(cfg, device="cpu"):
    """Instantiates M1-M5 models for matched comparative benchmarking and asserts used parameter matching."""
    # M1: Full WEFT Hybrid
    cfg_m1 = get_tiny_config(global_path="on", local_path="on")
    m1 = WeftForCausalLM(cfg_m1).to(device)
    target_params = sum(p.numel() for p in m1.parameters() if p.requires_grad)

    # Compute analytically matched intermediate sizes
    inter_transformer = compute_matching_intermediate_size(target_params, cfg, is_baseline=True)

    # M2: WEFT Local-Only (explicit config ablation flag)
    # Exclude unused state projections from trainable count by freezing & zeroing
    cfg_m2 = get_tiny_config(global_path="off", local_path="on", intermediate_size=inter_transformer)
    m2 = WeftForCausalLM(cfg_m2).to(device)
    for l in m2.model.layers:
        l.self_attn.proj_k.weight.requires_grad = False
        l.self_attn.proj_v.weight.requires_grad = False
        l.self_attn.decay_logits.requires_grad = False

    # M3: WEFT State-Only (explicit config ablation flag)
    # Exclude unused local projections from trainable count
    inter_m3 = round((target_params - (cfg.vocab_size * cfg.hidden_size + cfg.hidden_size + cfg.num_hidden_layers * (2 * cfg.hidden_size + 2 * (cfg.num_state_heads * cfg.state_rank * cfg.hidden_size) + cfg.num_state_heads * cfg.state_rank + cfg.hidden_size ** 2))) / (cfg.num_hidden_layers * 3 * cfg.hidden_size))
    cfg_m3 = get_tiny_config(global_path="on", local_path="off", intermediate_size=inter_m3)
    m3 = WeftForCausalLM(cfg_m3).to(device)
    for l in m3.model.layers:
        l.self_attn.q_proj.weight.requires_grad = False
        l.self_attn.k_proj.weight.requires_grad = False
        l.self_attn.v_proj.weight.requires_grad = False

    # M4: Transformer Full Attention (matched width/depth, RoPE, SwiGLU)
    m4 = BaselineTransformer(cfg, sliding_window=None, intermediate_size=inter_transformer).to(device)

    # M5: Transformer Sliding Window Only
    m5 = BaselineTransformer(cfg, sliding_window=cfg.window_size, intermediate_size=inter_transformer).to(device)

    models = {
        "M1_WEFT_Hybrid": m1,
        "M2_WEFT_LocalOnly": m2,
        "M3_WEFT_StateOnly": m3,
        "M4_Transformer_Full": m4,
        "M5_Transformer_Window": m5
    }

    # Verify and enforce used parameter matching within +-5% tolerance
    used_params = {
        k: sum(p.numel() for p in v.parameters() if p.requires_grad)
        for k, v in models.items()
    }
    total_params = {
        k: sum(p.numel() for p in v.parameters())
        for k, v in models.items()
    }

    ref_used = used_params["M1_WEFT_Hybrid"]
    for k, count in used_params.items():
        diff_pct = abs(count - ref_used) / ref_used * 100.0
        if diff_pct > 5.0:
            raise ValueError(
                f"Used parameter mismatch for {k}: {count} used params vs M1 {ref_used} ({diff_pct:.2f}% > 5.0% tolerance)"
            )

    return models, {"trainable": used_params, "total": total_params, "used": used_params}
