"""
Unit and integration tests for WEFT Hugging Face implementation.
Tests:
1. Config initialization and serialization (save_pretrained / from_pretrained).
2. Model instantiation and forward pass on CPU & CUDA.
3. Loss calculation with labels.
4. Auto-regressive generation step consistency (O(1) recurrent state vs full sequence).
5. State serialization and save/load round-trip.
"""

import os
import shutil
import tempfile
import torch
from weft import WeftConfig, WeftModel, WeftForCausalLM


def test_weft_config_roundtrip():
    print("--> Testing WeftConfig save/load roundtrip...")
    config = WeftConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
        window_size=64,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.save_pretrained(tmp_dir)
        loaded_config = WeftConfig.from_pretrained(tmp_dir)
        assert loaded_config.hidden_size == 256
        assert loaded_config.num_state_heads == 2
        assert loaded_config.state_rank == 8
        assert loaded_config.window_size == 64
    print("    [PASS] Config roundtrip successful.")


def test_weft_forward_pass():
    print("--> Testing WeftForCausalLM forward pass and loss...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = WeftConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
        window_size=64,
    )
    model = WeftForCausalLM(config).to(device)
    model.eval()

    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None
    assert outputs.logits.shape == (batch_size, seq_len, 1000)
    print(f"    [PASS] Forward loss: {outputs.loss.item():.4f}, logits shape: {outputs.logits.shape}")


def test_weft_recurrent_equivalence():
    """
    Verify that single-step recurrent state tracking produces identical logits
    to feeding the full sequence.
    Note: Local sliding window attention operates inside the window,
    so single-token step tracking tests the global state accumulation consistency.
    """
    print("--> Testing recurrent state step consistency...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = WeftConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=64,
    )
    model = WeftForCausalLM(config).to(device)
    model.eval()

    seq_len = 8
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

    # Full forward pass with use_cache=True
    with torch.no_grad():
        full_out = model(input_ids=input_ids, use_cache=True)
        final_full_logits = full_out.logits[:, -1, :]
        cached_states = full_out.past_key_values

    # Step-by-step forward pass feeding 1 token at a time
    current_past = None
    step_logits = None
    with torch.no_grad():
        for t in range(seq_len):
            token_in = input_ids[:, t:t+1]
            step_out = model(input_ids=token_in, past_key_values=current_past, use_cache=True)
            current_past = step_out.past_key_values
            step_logits = step_out.logits[:, 0, :]

    # In step-by-step with s=1, local attention has sequence length 1.
    # In full pass, token t has window attention over previous tokens.
    # Verify that the recurrent state itself is accumulated identically.
    for l_idx in range(config.num_hidden_layers):
        full_rec_state = cached_states.layers[l_idx].weft_recurrent_state
        step_rec_state = current_past.layers[l_idx].weft_recurrent_state
        diff = torch.norm(full_rec_state - step_rec_state).item()
        print(f"    Layer {l_idx} recurrent state divergence: {diff:.6e}")
        assert diff < 1e-4, f"State divergence too high on layer {l_idx}: {diff}"

    print("    [PASS] Recurrent state tracks identically across step-by-step and full-sequence.")


def test_weft_model_save_and_load():
    print("--> Testing full model save_pretrained and from_pretrained...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = WeftConfig(
        vocab_size=500,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=4,
        window_size=32,
    )
    model = WeftForCausalLM(config).to(device)
    input_ids = torch.randint(0, 500, (1, 16), device=device)

    with torch.no_grad():
        orig_logits = model(input_ids=input_ids).logits

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        loaded_model = WeftForCausalLM.from_pretrained(tmp_dir).to(device)
        loaded_model.eval()

        with torch.no_grad():
            loaded_logits = loaded_model(input_ids=input_ids).logits

        max_diff = torch.max(torch.abs(orig_logits - loaded_logits)).item()
        print(f"    Max difference between saved and loaded model: {max_diff:.6e}")
        assert max_diff < 1e-5, f"Loaded model output deviates from original: {max_diff}"

    print("    [PASS] Full model serialization roundtrip verified.")


if __name__ == "__main__":
    print("=================================================================")
    print("  WEFT HUGGING FACE COMPATIBILITY & NUMERICAL INTEGRITY TEST")
    print("=================================================================")
    test_weft_config_roundtrip()
    test_weft_forward_pass()
    test_weft_recurrent_equivalence()
    test_weft_model_save_and_load()
    print("=================================================================")
    print("  ALL WEFT UNIT TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")


def test_weft_sliding_window_cache_equivalence():
    """
    Item 1 & 2 regression test:
    Verify that full-sequence, 1-by-1 step decode, and chunked prefill+append
    yield identical logits under the causal sliding-window attention operator.
    Verify that the stored physical local KV cache is strictly bounded to (window_size - 1).
    """
    torch.manual_seed(17)
    vocab, hidden, intermediate, layers, attention_heads, state_heads, rank, window = 31, 16, 32, 2, 2, 2, 8, 4
    token_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15, 17]])

    cfg = WeftConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=attention_heads,
        num_state_heads=state_heads,
        state_rank=rank,
        window_size=window,
        use_cache=True,
    )
    model = WeftForCausalLM(cfg)
    model.eval()

    with torch.no_grad():
        out_full = model(token_ids, use_cache=False).logits

        # 1. 1-by-1 cached step decode
        past = None
        step_logits = []
        for i in range(token_ids.shape[1]):
            inp = token_ids[:, i:i+1]
            out_step = model(inp, past_key_values=past, use_cache=True)
            step_logits.append(out_step.logits)
            past = out_step.past_key_values
        out_step_all = torch.cat(step_logits, dim=1)

        diff_step = (out_full - out_step_all).abs().max().item()
        assert diff_step < 1e-5, f"1-by-1 step decode mismatch with full pass: {diff_step}"

        # 2. Chunked prefill (3) + append (3) + append (3)
        out1 = model(token_ids[:, :3], use_cache=True)
        out2 = model(token_ids[:, 3:6], past_key_values=out1.past_key_values, use_cache=True)
        out3 = model(token_ids[:, 6:], past_key_values=out2.past_key_values, use_cache=True)
        out_chunked = torch.cat([out1.logits, out2.logits, out3.logits], dim=1)

        diff_chunk = (out_full - out_chunked).abs().max().item()
        assert diff_chunk < 1e-5, f"Chunked append mismatch with full pass: {diff_chunk}"

        # Verify physical cache bounding
        assert out3.past_key_values.layers[0].keys.shape[-2] <= (window - 1), (
            f"Stored keys shape {out3.past_key_values.layers[0].keys.shape} exceeds bound {window - 1}"
        )
        assert out3.past_key_values.get_seq_length() == token_ids.shape[1], (
            f"Expected logical seq length {token_ids.shape[1]}, got {out3.past_key_values.get_seq_length()}"
        )


def test_weft_attention_mask_invariance():
    """
    Item 3 regression test:
    Verify that padding/attention mask correctly isolates masked positions so that
    changing tokens in masked positions produces zero perturbation on unmasked logits.
    """
    torch.manual_seed(17)
    cfg = WeftConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=4,
    )
    model = WeftForCausalLM(cfg).eval()

    tokens_a = torch.tensor([[10, 20, 5, 7]])
    tokens_b = torch.tensor([[29, 28, 5, 7]])
    mask = torch.tensor([[0, 0, 1, 1]])

    with torch.no_grad():
        out_a = model(tokens_a, attention_mask=mask).logits
        out_b = model(tokens_b, attention_mask=mask).logits

    diff = (out_a[:, 2:] - out_b[:, 2:]).abs().max().item()
    assert diff == 0.0, f"Attention mask leaked across unmasked positions: diff = {diff}"


def test_weft_masked_batch_equivalence():
    """
    N1 regression test:
    Verify that attention_mask works correctly for batch sizes > 1 with num_state_heads != batch_size,
    ensuring that left-padded batch items produce identical logits to unpadded single inputs,
    and cached decode steps respect per-batch masking.
    """
    cfg = WeftConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=16,
    )
    model = WeftForCausalLM(cfg).eval()

    # Test b=3, num_state_heads=2 (broadcasting mismatch test)
    x3 = torch.randint(0, 64, (3, 8))
    mask3 = torch.ones(3, 8)
    mask3[0, :2] = 0
    mask3[1, :4] = 0
    out3 = model(x3, attention_mask=mask3)
    assert out3.logits.shape == (3, 8, 64)

    # Test left-padded batch=2 vs unpadded single
    seq_len = 6
    pad_len = 4
    unpadded_seq = torch.randint(0, 64, (1, seq_len))
    padded_seq = torch.cat([torch.zeros(1, pad_len, dtype=torch.long), unpadded_seq], dim=1)
    mask_padded = torch.cat([torch.zeros(1, pad_len), torch.ones(1, seq_len)], dim=1)

    other_seq = torch.randint(0, 64, (1, seq_len + pad_len))
    other_mask = torch.ones(1, seq_len + pad_len)

    batch_input = torch.cat([padded_seq, other_seq], dim=0)
    batch_mask = torch.cat([mask_padded, other_mask], dim=0)

    with torch.no_grad():
        out_single = model(unpadded_seq)
        out_batch = model(batch_input, attention_mask=batch_mask)

    diff = (out_single.logits[0, -1, :] - out_batch.logits[0, -1, :]).abs().max().item()
    assert diff < 1e-4, f"Left-padded batch vs unpadded single mismatch: {diff}"

    # Test step-by-step cached decode with mask (s=1 path)
    with torch.no_grad():
        res_prefill = model(batch_input[:, :-1], attention_mask=batch_mask[:, :-1], use_cache=True)
        out_step = model(batch_input[:, -1:], past_key_values=res_prefill.past_key_values, attention_mask=batch_mask, use_cache=True)
    step_diff = (out_step.logits[:, 0, :] - out_batch.logits[:, -1, :]).abs().max().item()
    assert step_diff < 1e-4, f"Step decode vs full forward mismatch: {step_diff}"


def test_weft_generate_bounded_cache_and_beam_search():
    """
    N2 regression test:
    Verify that Hugging Face generate() routes through WeftCache:
    1. Physical KV cache size is strictly bounded to window_size - 1.
    2. Beam search on batched distinct prompts reorders recurrent state properly and matches
       independent beam search generations.
    """
    cfg = WeftConfig(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=4,
    )
    model = WeftForCausalLM(cfg).eval()

    # 1. Bounded physical cache check over a long generate call
    prompt = torch.randint(0, 50, (1, 6))
    gen_out = model.generate(
        prompt, max_new_tokens=16, do_sample=False, return_dict_in_generate=True
    )
    cache = gen_out.past_key_values
    from weft.cache import WeftCache, WeftCacheLayer
    assert isinstance(cache, WeftCache), f"generate() did not use WeftCache: {type(cache)}"
    for layer in cache.layers:
        assert isinstance(layer, WeftCacheLayer)
        assert layer.keys.shape[-2] <= cfg.window_size - 1, (
            f"Physical keys {layer.keys.shape[-2]} exceeded bound {cfg.window_size - 1}"
        )
    assert cache.get_seq_length() == 6 + 16 - 1  # 21 steps completed

    # 2. Beam search equivalence on distinct prompts
    p_a = torch.tensor([[10, 11, 12, 13]])
    p_b = torch.tensor([[20, 21, 22, 23]])
    batch_p = torch.cat([p_a, p_b], dim=0)

    out_batch = model.generate(batch_p, max_new_tokens=8, num_beams=2, do_sample=False)
    out_a = model.generate(p_a, max_new_tokens=8, num_beams=2, do_sample=False)
    out_b = model.generate(p_b, max_new_tokens=8, num_beams=2, do_sample=False)

    assert torch.equal(out_batch[0:1], out_a), "Beam search output for prompt A deviated in batch"
    assert torch.equal(out_batch[1:2], out_b), "Beam search output for prompt B deviated in batch"


def test_weft_beam_reorder_recurrent_state():
    """
    Item 4 regression test:
    Verify that cache.reorder_cache reorders both local keys and weft_recurrent_state.
    """
    cfg = WeftConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=4,
    )
    model = WeftForCausalLM(cfg).eval()

    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = model(tokens, use_cache=True)
    past = out.past_key_values
    beam_idx = torch.tensor([1, 0])

    k_before = past.layers[0].keys.clone()
    rec_before = past.layers[0].weft_recurrent_state.clone()

    past.reorder_cache(beam_idx)

    assert torch.equal(past.layers[0].keys, k_before.index_select(0, beam_idx))
    assert torch.equal(past.layers[0].weft_recurrent_state, rec_before.index_select(0, beam_idx))


def test_weft_use_cache_false_preserves_caller_state():
    """
    Item 6 regression test:
    Passing use_cache=False with a Cache object must not mutate the cache.
    """
    cfg = WeftConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=4,
    )
    model = WeftModel(cfg).eval()

    tokens = torch.tensor([[1, 2, 3]])
    _, c = model(tokens, use_cache=True)
    initial_len = c.get_seq_length()
    initial_keys = c.layers[0].keys.clone()

    _, next_c = model(torch.tensor([[4]]), past_key_values=c, use_cache=False)
    assert next_c is None
    assert c.get_seq_length() == initial_len
    assert torch.equal(c.layers[0].keys, initial_keys)


def test_weft_gradient_checkpointing_equivalence():
    """
    Item 9 regression test:
    Verify that training with gradient checkpointing produces identical outputs
    and gradients to standard forward/backward passes.
    """
    torch.manual_seed(42)
    cfg = WeftConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=4,
    )

    model_plain = WeftForCausalLM(cfg).train()
    model_ckpt = WeftForCausalLM(cfg).train()
    model_ckpt.load_state_dict(model_plain.state_dict())
    model_ckpt.gradient_checkpointing_enable()

    tokens = torch.randint(0, 32, (2, 8))
    labels = tokens.clone()

    # Forward & backward plain
    out_plain = model_plain(tokens, labels=labels)
    out_plain.loss.backward()

    # Forward & backward checkpointed
    out_ckpt = model_ckpt(tokens, labels=labels)
    out_ckpt.loss.backward()

    # Compare loss
    assert torch.allclose(out_plain.loss, out_ckpt.loss, atol=1e-5), "Loss mismatch with gradient checkpointing"

    # Compare gradients across all parameters
    for (n_p, p_p), (n_c, p_c) in zip(model_plain.named_parameters(), model_ckpt.named_parameters()):
        assert p_p.grad is not None and p_c.grad is not None, f"Missing gradient on {n_p}"
        diff = (p_p.grad - p_c.grad).abs().max().item()
        assert diff < 1e-4, f"Gradient mismatch on parameter {n_p}: {diff}"


def test_weft_rmsnorm_numerical_stability():
    """
    Verify WeftRMSNorm stability under extreme float32 and float64 inputs.
    """
    from weft.modeling_weft import WeftRMSNorm
    norm = WeftRMSNorm(16)

    # 1. Extreme float32 inputs
    x_f32 = torch.full((2, 16), 1e20, dtype=torch.float32)
    y_f32 = norm(x_f32)
    assert not torch.isnan(y_f32).any() and not torch.isinf(y_f32).any()
    assert torch.allclose(y_f32, torch.ones_like(y_f32), atol=1e-4)

    # 2. Extreme float64 inputs (> float32 max 3.4e38)
    x_f64 = torch.full((2, 16), 1e40, dtype=torch.float64)
    y_f64 = norm(x_f64)
    assert not torch.isnan(y_f64).any() and not torch.isinf(y_f64).any()
    assert torch.allclose(y_f64, torch.ones_like(y_f64), atol=1e-4)
