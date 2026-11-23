"""
Unit tests for Authentic Open Conjectures and Conditional Reductions in Lean 4.
"""

import subprocess
import pytest
from pathlib import Path
from weft.conditional_explorer import ConditionalExplorerEngine, AUTHENTIC_CONJECTURES


def test_lean_conditional_conjectures_compiles():
    """Verify formal/real_conjectures/ConditionalConjectures.lean compiles cleanly with zero sorry."""
    lean_file = Path("formal/real_conjectures/ConditionalConjectures.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in ConditionalConjectures.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in ConditionalConjectures.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_conditional_explorer_catalog():
    """Verify authentic open conjecture metadata and formal statements."""
    engine = ConditionalExplorerEngine()
    assert len(engine.conjectures) == 5
    for c in engine.conjectures:
        assert c.conjecture_id.startswith("AOC-")
        assert len(c.conditional_reduction) > 0
        assert len(c.lean_theorem) > 0


def test_bounded_collatz_scanner():
    """Verify bounded finite-range Collatz counterexample search."""
    engine = ConditionalExplorerEngine()
    res = engine.run_bounded_collatz_scan(10000)
    assert res["counterexamples_found"] == 0
    assert res["all_reached_one"] is True
    assert res["scan_latency_ms"] < 2000.0


def test_conditional_explorer_full_run():
    """Run full evaluation suite."""
    engine = ConditionalExplorerEngine()
    reports = engine.evaluate_all()
    assert len(reports) == 5
    for r in reports:
        assert r["lean_verified"] is True
        assert r["eval_ms"] < 1000.0
