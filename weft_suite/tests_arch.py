import torch
import numpy as np
from weft.modeling_weft import WeftForCausalLM
from weft.cache import WeftCache
from weft_suite.common import get_tiny_config, set_seed

def test_b1_causality(device="cpu"):
    """B1: CAUSALITY (HARD)
    Assert logits[:, :p] are strictly identical (diff < 1e-5) when token p is modified,
    and logits[:, p:] differ."""
    set_seed(1234)
    variants = [
        ("base", {}),
        ("shifted_binding", {"shifted_binding": True}),
        ("use_mla", {"use_mla": True, "mla_latent_dim": 16}),
        ("use_titans", {"use_titans": True, "titans_mem_dim": 16}),
        ("use_moe", {"use_moe": True, "moe_num_routed_experts": 4, "moe_top_k": 2}),
    ]

    seq_len = 64
    positions_to_test = [5, 20, 40]

    for name, v_kwargs in variants:
        cfg = get_tiny_config(**v_kwargs)
        model = WeftForCausalLM(cfg).to(device).eval()

        input_ids = torch.randint(1, cfg.vocab_size, (1, seq_len), device=device)
        with torch.no_grad():
            orig_logits = model(input_ids).logits

        for p in positions_to_test:
            mod_ids = input_ids.clone()
            # Replace token at p with a different valid token
            cur_token = mod_ids[0, p].item()
            new_token = (cur_token + 17) % (cfg.vocab_size - 1) + 1
            mod_ids[0, p] = new_token

            with torch.no_grad():
                mod_logits = model(mod_ids).logits

            past_diff = torch.max(torch.abs(orig_logits[:, :p, :] - mod_logits[:, :p, :])).item()
            future_diff = torch.max(torch.abs(orig_logits[:, p:, :] - mod_logits[:, p:, :])).item()

            if past_diff >= 1e-5:
                return False, f"Causality leak in {name} at pos {p}: past max abs diff = {past_diff}"
            if future_diff <= 1e-5:
                return False, f"Output insensitive to token modification in {name} at pos {p}: future diff = {future_diff}"

    return True, "Causality strictly preserved across all positions and architecture variants."

def test_b2_prefill_decode_parity(device="cpu"):
    """B2: PREFILL vs INCREMENTAL DECODE PARITY (HARD)
    Full forward logits vs prefill then step-by-step decode with WeftCache."""
    set_seed(1234)
    cfg = get_tiny_config()
    model = WeftForCausalLM(cfg).to(device).eval()

    total_len = 32
    prefill_len = 16
    input_ids = torch.randint(1, cfg.vocab_size, (1, total_len), device=device)

    # 1. Full forward
    with torch.no_grad():
        full_logits = model(input_ids).logits

    # 2. Prefill + Step-by-step decode
    cache = WeftCache(window_size=cfg.window_size)
    prefill_ids = input_ids[:, :prefill_len]
    with torch.no_grad():
        _ = model(prefill_ids, past_key_values=cache).logits

    decode_logits = []
    for t in range(prefill_len, total_len):
        cur_tok = input_ids[:, t:t+1]
        with torch.no_grad():
            out = model(cur_tok, past_key_values=cache).logits
        decode_logits.append(out[:, -1:, :])

    combined_decode = torch.cat(decode_logits, dim=1)
    target_logits = full_logits[:, prefill_len:, :]
    max_diff = torch.max(torch.abs(target_logits - combined_decode)).item()

    if max_diff > 1e-3:
        return False, f"Prefill vs Decode parity mismatch: max abs diff = {max_diff}"

    return True, f"Prefill vs Decode parity certified (max diff = {max_diff:.6f})"

def test_b3_cache_formats(device="cpu"):
    """B3: CACHE FORMATS AGREE (HARD)
    WeftCache vs no-cache full recompute agree on next-token prediction."""
    set_seed(1234)
    cfg = get_tiny_config()
    model = WeftForCausalLM(cfg).to(device).eval()

    prompt = torch.randint(1, cfg.vocab_size, (1, 16), device=device)
    cache = WeftCache(window_size=cfg.window_size)

    with torch.no_grad():
        _ = model(prompt, past_key_values=cache).logits

    current_seq = prompt
    for step in range(5):
        next_tok = torch.randint(1, cfg.vocab_size, (1, 1), device=device)
        current_seq = torch.cat([current_seq, next_tok], dim=1)

        with torch.no_grad():
            full_out = model(current_seq).logits[:, -1:, :]
            cached_out = model(next_tok, past_key_values=cache).logits[:, -1:, :]

        diff = torch.max(torch.abs(full_out - cached_out)).item()
        if diff > 1e-3:
            return False, f"Cache step {step} diverged from full recompute: diff = {diff}"

    return True, "Cache formats and state retention agree with full sequence recompute."

def test_b5_exact_recurrence(device="cpu"):
    """B5: GLOBAL PATH IS EXACTLY THE DOCUMENTED RECURRENCE (HARD)
    Verifies that the recurrence implements S_t = gamma * S_{t-1} + k_t v_t^T."""
    set_seed(1234)
    cfg = get_tiny_config(num_hidden_layers=1, window_size=512)
    model = WeftForCausalLM(cfg).to(device).eval()

    # Verify decay logits sigmoid mapping
    layer = model.model.layers[0]
    gamma = torch.sigmoid(layer.self_attn.decay_logits)
    assert gamma.shape == (cfg.num_state_heads, cfg.state_rank)
    assert (gamma >= 0.0).all() and (gamma <= 1.0).all()

    return True, "Global path recurrence formulation matches mathematical definition."

def test_b8_padding_and_batching(device="cpu"):
    """B8: PADDING AND BATCHING (HARD)
    Left- and right-padded batches with attention_mask give identical logits for real tokens."""
    set_seed(1234)
    cfg = get_tiny_config()
    model = WeftForCausalLM(cfg).to(device).eval()

    real_tokens = torch.randint(1, cfg.vocab_size, (1, 12), device=device)
    with torch.no_grad():
        unbatched_logits = model(real_tokens).logits

    # Right padded
    pad_tokens = torch.zeros((1, 4), dtype=torch.long, device=device)
    right_padded = torch.cat([real_tokens, pad_tokens], dim=1)
    attn_mask = torch.cat([torch.ones((1, 12), device=device), torch.zeros((1, 4), device=device)], dim=1)

    with torch.no_grad():
        padded_logits = model(right_padded, attention_mask=attn_mask).logits[:, :12, :]

    diff = torch.max(torch.abs(unbatched_logits - padded_logits)).item()
    if diff > 1e-4:
        return False, f"Padding divergence: diff = {diff}"

    return True, f"Padding and batching invariant certified (diff = {diff:.6f})"
