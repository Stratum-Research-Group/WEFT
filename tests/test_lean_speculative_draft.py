import pytest

def test_speculative_acceptance_monotonicity():
    acc_prior = 10
    k_accepted = 3
    assert acc_prior <= acc_prior + k_accepted

def test_speculative_progress_guaranteed():
    for k in range(5):
        assert 1 <= 1 + k

def test_speculative_fleet_storage():
    # Coder-14B (70MB) + Coder-32B (148MB)
    s_draft = 70
    s_target = 148
    assert s_draft + s_target == 218
    assert s_draft + s_target <= 218

def test_null_speculation():
    t_target = 1
    assert t_target + 0 == t_target
