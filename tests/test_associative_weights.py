"""
Unit and benchmark tests for WeftAssociativeWeight and low-memory optimizers.
"""

import torch
import pytest
from weft.associative_weight import WeftAssociativeWeight
from weft.optimizers import Quantized8BitAdamW, Muon


def test_associative_weight_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, s, d = 2, 64, 128
    x = torch.randn(b, s, d, device=device, requires_grad=True)
    amw = WeftAssociativeWeight(hidden_size=d, num_memory_slots=256, rank=64).to(device)

    out = amw(x)
    assert out.shape == (b, s, d)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert amw.memory_keys.grad is not None
    assert amw.memory_values.grad is not None
    assert not torch.isnan(x.grad).any()


def test_quantized_8bit_adamw_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(64, 64, device=device, requires_grad=True)
    opt = Quantized8BitAdamW([w], lr=1e-3)

    initial_loss = (w ** 2).sum().item()
    for _ in range(5):
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    final_loss = (w ** 2).sum().item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_muon_optimizer_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(64, 64, device=device, requires_grad=True)
    opt = Muon([w], lr=0.01)

    initial_loss = (w ** 2).sum().item()
    for _ in range(5):
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    final_loss = (w ** 2).sum().item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
