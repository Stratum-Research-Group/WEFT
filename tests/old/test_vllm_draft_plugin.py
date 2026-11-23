"""Unit tests for vLLM / SGLang speculative proposal plugin."""
import torch
import pytest
from weft.vllm_draft_plugin import WeftVLLMDraftPlugin

def test_vllm_draft_plugin_propose():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plugin = WeftVLLMDraftPlugin(device=device, gamma=4, vocab_size=64, student_dim=64, teacher_dim=128)
    
    prefix = torch.randint(0, 64, (2, 8), device=device)
    teacher_embed = torch.randn(64, 128, device=device)
    
    draft_tokens, draft_probs = plugin.propose_draft_tokens(prefix, teacher_embed)
    
    assert draft_tokens.shape == (2, 4), f"Expected [2, 4], got {draft_tokens.shape}"
    assert draft_probs.shape == (2, 4, 64), f"Expected [2, 4, 64], got {draft_probs.shape}"
    assert (draft_tokens >= 0).all() and (draft_tokens < 64).all()

def test_vllm_draft_plugin_config():
    plugin = WeftVLLMDraftPlugin(gamma=4)
    cfg = plugin.export_vllm_config()
    assert cfg["model_type"] == "weft_sula_speculative_draft"
    assert cfg["num_speculative_tokens"] == 4
    assert cfg["speedup_factor"] == 2.08
