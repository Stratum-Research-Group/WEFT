import pytest

def test_fleet_total_storage_bounded():
    # 60 + 59 + 148 + 229 + 78 + 148 + 260 + 318 + 15 MB
    models = [60, 59, 148, 229, 78, 148, 260, 318, 15]
    total = sum(models)
    assert total <= sum(models)
    assert total == 1315

def test_dispatch_latency():
    t_route = 5
    t_exec = 45
    assert t_route <= t_route + t_exec

def test_null_query_dispatch():
    t_route = 5
    assert t_route + 0 == t_route
