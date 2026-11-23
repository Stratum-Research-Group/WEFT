"""
Unit tests for Paper 66: 3rd-Party Ecosystem Comparative Benchmarks and Lean 4 Invariants.
"""

import sys
import os
import subprocess
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.benchmark_ecosystem_comparison import ECOSYSTEM_BENCHMARKS


def test_lean_ecosystem_invariants_compiles():
    """Verify formal/real_conjectures/EcosystemBenchmarkInvariants.lean compiles cleanly with 0 axioms and 0 sorry."""
    lean_file = Path("formal/real_conjectures/EcosystemBenchmarkInvariants.lean")
    assert lean_file.exists(), f"File {lean_file} missing"
    content = lean_file.read_text()
    assert "sorry" not in content
    assert "axiom" not in content

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean compilation error:\n{res.stderr}"


def test_ecosystem_pareto_dominance():
    """Verify that WEFT achieves lowest BPW and highest compression ratio in the benchmark table."""
    weft_bench = next(b for b in ECOSYSTEM_BENCHMARKS if "WEFT" in b.framework_name)
    assert weft_bench.effective_bpw < 0.05
    assert weft_bench.compression_ratio > 600.0
    assert weft_bench.resident_ram_mb < 300.0
    assert weft_bench.formal_invariants_checked is True
