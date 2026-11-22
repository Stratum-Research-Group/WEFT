"""Test suite verifying formal Lean 4 MoE Routing Invariants."""

import subprocess
import shutil
import pytest
from pathlib import Path

LEAN_FILE = Path(__file__).parent.parent / "formal" / "MoERoutingInvariants.lean"


def test_lean_moe_routing_file_exists():
    assert LEAN_FILE.exists(), f"Lean proof file missing: {LEAN_FILE}"


@pytest.mark.skipif(shutil.which("lean") is None, reason="Lean 4 compiler not installed")
def test_lean_moe_routing_theorems_compile():
    """Verify that Lean 4 formally compiles and checks all theorems in MoERoutingInvariants.lean."""
    res = subprocess.run(
        ["lean", str(LEAN_FILE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0, f"Lean verification failed with error:\n{res.stderr}\n{res.stdout}"


def test_moe_routing_theorems_present():
    """Verify that all required theorems are written into the proof file."""
    content = LEAN_FILE.read_text(encoding="utf-8")
    expected_theorems = [
        "total_dispatch_comm",
        "total_dispatch_zero_k",
        "total_dispatch_mono_k",
        "top_k_dispatch_bounded_by_total",
        "total_dispatch_zero_tokens",
        "total_dispatch_add_tokens",
    ]
    for thm in expected_theorems:
        assert f"theorem {thm}" in content, f"Expected theorem '{thm}' not declared in {LEAN_FILE.name}"
