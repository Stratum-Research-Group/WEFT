"""
Unit tests for Paper 65: Universal Adapter Knowledge Saturation and Lean 4 Invariants.
"""

import subprocess
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from weft.api_server import app
from weft.adapter_store import PersistentAdapterStore


def test_lean_adapter_saturation_invariants_compiles():
    """Verify formal/real_conjectures/AdapterSaturationInvariants.lean compiles cleanly with 0 axioms and 0 sorry."""
    lean_file = Path("formal/real_conjectures/AdapterSaturationInvariants.lean")
    assert lean_file.exists(), f"File {lean_file} missing"
    content = lean_file.read_text()
    assert "sorry" not in content
    assert "axiom" not in content

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean compilation error:\n{res.stderr}"


def test_160_adapters_saturation_cardinality():
    """Verify all 160 Frontier problems have persisted adapters."""
    store = PersistentAdapterStore(store_dir="checkpoints/adapters")
    adapters = store.list_retained_adapters()
    f160_adapters = [a for a in adapters if a["conjecture_id"].startswith("F160-")]
    assert len(f160_adapters) == 160


def test_api_server_adapter_hot_loading():
    """Verify runtime hot-loading via FastAPI endpoint."""
    client = TestClient(app)
    res = client.post("/v1/adapters/load/ADAPT-F160-001")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "LOADED_IN_REGISTER"
    assert data["shape"] == [4096, 4096]
    assert data["certified_lean"] is True
