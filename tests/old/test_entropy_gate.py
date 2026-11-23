"""Unit tests for Lean 4 verified Adaptive Entropy Gate."""
import torch
import pytest
from weft.entropy_gate import AdaptiveSpeculativeEntropyGate

def test_entropy_gate_decisions():
    gate = AdaptiveSpeculativeEntropyGate(gamma=4, entropy_threshold=5.0)

    # 1. Low entropy (one-hot confident logits) -> Should speculate
    low_ent_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    allow, ent = gate.should_speculate(low_ent_logits)
    assert allow is True
    assert ent < 1.0

    # 2. High entropy (uniform distribution over 1024 classes) -> Should block speculation
    high_ent_logits = torch.zeros((1, 1024))
    allow, ent = gate.should_speculate(high_ent_logits)
    assert allow is False
    assert ent >= 9.9 # log2(1024) = 10 bits

def test_provable_speedup_lower_bound():
    gate = AdaptiveSpeculativeEntropyGate(gamma=4)
    # If alpha is 0.65 -> Speedup > 2.0x
    assert gate.compute_provable_speedup(0.65) > 2.0
    # If alpha drops below break-even (e.g. 0.05) -> Speedup is clamped to 1.0 (no slowdown!)
    assert gate.compute_provable_speedup(0.05) == 1.0
