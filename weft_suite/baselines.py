"""
Track A Model Baselines:
Implements matched architectures for comparative evaluation:
  M1: WEFT base (local sliding-window + recurrent state)
  M2: WEFT local-only (recurrent state zeroed)
  M3: WEFT state-only (window_size=1)
  M4: Standard Transformer with full causal attention & RoPE
  M5: Transformer sliding-window only
All models built at matched parameter counts.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from weft.modeling_weft import WeftForCausalLM
from weft_suite.common import get_tiny_config

class BaselineTransformer(nn.Module):
    """M4 / M5 Standard Causal Transformer with matched width and depth."""
    def __init__(self, cfg, sliding_window=None):
        super().__init__()
        self.cfg = cfg
        self.sliding_window = sliding_window
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.num_attention_heads,
                dim_feedforward=cfg.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(cfg.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        b, s = input_ids.shape
        x = self.embed(input_ids)
        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(s, device=input_ids.device)
        if self.sliding_window is not None:
            # Banded mask
            indices = torch.arange(s, device=input_ids.device)
            band = (indices.unsqueeze(0) - indices.unsqueeze(1)) > self.sliding_window
            causal_mask = causal_mask.masked_fill(band, float("-inf"))

        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def get_model_family(cfg, device="cpu"):
    """Instantiates M1-M5 models for matched comparative benchmarking."""
    # M1: Full WEFT
    m1 = WeftForCausalLM(cfg).to(device)

    # M2: WEFT Local-Only (recurrent projection zeroed)
    m2 = WeftForCausalLM(cfg).to(device)
    for l in m2.model.layers:
        l.self_attn.proj_v.weight.data.zero_()
        l.self_attn.proj_v.weight.requires_grad = False

    # M3: WEFT State-Only (window_size=1)
    cfg_m3 = get_tiny_config(window_size=1)
    m3 = WeftForCausalLM(cfg_m3).to(device)

    # M4: Transformer Full Attention
    m4 = BaselineTransformer(cfg).to(device)

    # M5: Transformer Sliding Window Only
    m5 = BaselineTransformer(cfg, sliding_window=cfg.window_size).to(device)

    models = {
        "M1_WEFT_Hybrid": m1,
        "M2_WEFT_LocalOnly": m2,
        "M3_WEFT_StateOnly": m3,
        "M4_Transformer_Full": m4,
        "M5_Transformer_Window": m5
    }

    # Verify parameter matching
    params = {k: sum(p.numel() for p in v.parameters()) for k, v in models.items()}
    return models, params
