"""Unit tests for Multi-Token Prediction (MTP) Head."""
import torch
import pytest
from weft.mtp_head import MultiTokenPredictionHead

def test_mtp_parallel_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtp = MultiTokenPredictionHead(student_dim=64, teacher_dim=128, num_future_tokens=4).to(device)
    embed_w = torch.randn(32, 128, device=device)
    h_last = torch.randn(2, 64, device=device)

    # 1. Forward test
    logits = mtp(h_last, embed_w)
    assert logits.shape == (2, 4, 32), f"Expected [2, 4, 32], got {logits.shape}"

    # 2. Parallel single-shot candidate drafting
    candidates = mtp.predict_parallel_draft_chain(h_last, embed_w)
    assert candidates.shape == (2, 4), f"Expected [2, 4], got {candidates.shape}"
    assert (candidates >= 0).all() and (candidates < 32).all()
