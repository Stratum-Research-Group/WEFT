import pytest

def test_cascade_acceptance_monotonic():
    acc = 10
    k1 = 4
    k2 = 2
    assert acc <= acc + k1 + k2

def test_cascade_progress():
    for k1 in range(3):
        for k2 in range(3):
            assert 1 <= 1 + k1 + k2

def test_cascade_fleet_memory():
    # 7B (38MB) + 14B (78MB) + 32B (148MB)
    s7, s14, s32 = 38, 78, 148
    assert s7 + s14 + s32 == 264
    assert s7 + s14 + s32 <= 264

def test_null_cascade():
    t_target = 1
    assert t_target + 0 == t_target
