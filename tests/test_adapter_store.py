"""
Unit tests for Persistent Adapter Store and knowledge retention.
"""

import torch
import pytest
from pathlib import Path
from weft.adapter_store import PersistentAdapterStore


def test_persistent_adapter_roundtrip():
    """Verify saving and reloading adapters preserves exact tensor weights."""
    store = PersistentAdapterStore(store_dir="checkpoints/adapters_test")
    d = 128
    u = torch.randn(d, 1)
    v = torch.randn(1, d)
    expected_delta = u @ v

    p = store.save_adapter(
        adapter_id="ADAPT-UNIT-001",
        target_layer="test.layer.0",
        domain="Unit Testing",
        conjecture_id="TEST-001",
        u=u, v=v
    )
    assert p.exists()

    loaded = store.load_adapter("ADAPT-UNIT-001")
    assert loaded is not None
    assert torch.allclose(loaded, expected_delta, atol=1e-6)

    # Clean up test directory
    for f in Path("checkpoints/adapters_test").glob("*"):
        f.unlink()
    Path("checkpoints/adapters_test").rmdir()
