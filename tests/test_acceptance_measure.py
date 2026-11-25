import torch
import pytest

from weft.acceptance_measure import accepted_prefix_len, measure_acceptance


def _logits_for(tokens, vocab=8):
    out = torch.zeros(len(tokens), vocab)
    for i, t in enumerate(tokens):
        out[i, t] = 5.0
    return out


def test_full_acceptance():
    d = torch.tensor([1, 2, 3, 4])
    assert accepted_prefix_len(d, _logits_for([1, 2, 3, 4])) == 4


def test_prefix_stops_at_first_mismatch():
    d = torch.tensor([1, 2, 3, 4])
    # target disagrees at index 2; index 3 matches but must NOT be counted
    assert accepted_prefix_len(d, _logits_for([1, 2, 7, 4])) == 2


def test_zero_acceptance():
    d = torch.tensor([1, 2])
    assert accepted_prefix_len(d, _logits_for([0, 0])) == 0


def test_measure_acceptance_rate_and_histogram():
    rounds = [
        {"draft_tokens": torch.tensor([1, 2, 3, 4]), "target_logits": _logits_for([1, 2, 3, 4])},
        {"draft_tokens": torch.tensor([1, 2, 3, 4]), "target_logits": _logits_for([1, 7, 3, 4])},
    ]
    m = measure_acceptance(rounds)
    assert m["tokens_drafted"] == 8
    assert m["tokens_accepted"] == 5
    assert m["acceptance_rate"] == pytest.approx(5 / 8)
    assert m["accepted_prefix_histogram"] == {1: 1, 4: 1}


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        accepted_prefix_len(torch.tensor([1, 2]), _logits_for([1, 2, 3]))
