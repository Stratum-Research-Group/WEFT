"""Unit tests for WeftCUDAGraphRunner."""
import torch
import pytest
from weft import WeftConfig, WeftForCausalLM
from weft.cuda_graph_runner import WeftCUDAGraphRunner

class DummyBridge(torch.nn.Module):
    def __init__(self, in_d=64, out_d=128):
        super().__init__()
        self.fc = torch.nn.Linear(in_d, out_d, bias=False)
    def forward(self, h, embed_w):
        return self.fc(h) @ embed_w.t()

def test_cuda_graph_runner_replay():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WeftConfig(vocab_size=32, hidden_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=32)
    model = WeftForCausalLM(cfg).to(device)
    bridge = DummyBridge(64, 128).to(device)
    embed_w = torch.randn(32, 128, device=device)

    runner = WeftCUDAGraphRunner(
        draft_model=model,
        bridge=bridge,
        teacher_embed_w=embed_w,
        batch_size=1,
        seq_len=16,
        device=device,
    )

    test_ids = torch.randint(0, 32, (1, 16), device=device)
    logits = runner.replay_forward(test_ids)

    assert logits.shape == (1, 32)
    assert not torch.isnan(logits).any()
