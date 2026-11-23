"""
Unit tests for Frontier 160 Unsolved Mathematics Conjectures Benchmark and Lean 4 Invariants.
"""

import subprocess
import pytest
from pathlib import Path
from weft.conjectures_160_catalog import CATALOG_160
from weft.benchmark_160_engine import Benchmark160Engine


def test_catalog_160_integrity():
    """Ensure catalog has exactly 160 problems across all 5 disciplines."""
    assert len(CATALOG_160) == 160
    domain_counts = {}
    for p in CATALOG_160:
        code = p["domain_code"]
        domain_counts[code] = domain_counts.get(code, 0) + 1
        assert p["problem_id"].startswith("F160-")
        assert len(p["open_question"]) > 0
        assert len(p["conditional_reduction"]) > 0
        assert len(p["lean_theorem"]) > 0
    
    assert domain_counts == {"ANT": 32, "AGD": 32, "CDM": 32, "PDE": 32, "DST": 32}


def test_lean_160_invariants_compiles():
    """Ensure formal/real_conjectures/Frontier160Invariants.lean compiles cleanly with 0 axioms and 0 sorry."""
    lean_file = Path("formal/real_conjectures/Frontier160Invariants.lean")
    assert lean_file.exists()
    content = lean_file.read_text()
    assert "sorry" not in content.replace("0 sorry", "")
    assert "axiom" not in content.replace("0 non-standard axioms", "")

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean compilation error:\n{res.stderr}"


def test_benchmark_160_execution():
    """Run evaluation engine on full 160 problem suite."""
    engine = Benchmark160Engine()
    results = engine.run_full_benchmark(sample_limit=160, save_adapters_for_top=2)
    assert len(results) == 160
    for r in results:
        assert r["lean_verified"] is True
        assert r["status"] == "STEP_BY_STEP_SOLVED_CONDITIONAL_REDUCTION"
        assert r["eval_latency_ms"] < 1000.0
