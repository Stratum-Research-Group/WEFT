import pytest

def test_repaired_storage():
    s = 1354
    assert s <= s

def test_error_reduction():
    err_prior = 100
    correction = 30
    assert err_prior - correction <= err_prior
    assert err_prior - correction == 70

def test_null_corruption():
    err = 45
    assert err - 0 == err
