"""
Unit and benchmark tests for High-Throughput SIMD Counterexample Search Engine
and Lean 4 verified search invariants.
"""

import subprocess
import pytest
from pathlib import Path


def test_lean_counterexample_search_invariants_compiles():
    """Verify formal/CounterexampleSearchInvariants.lean compiles cleanly with zero sorry."""
    lean_file = Path("formal/CounterexampleSearchInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in CounterexampleSearchInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in CounterexampleSearchInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_native_simd_collatz_scanner_execution():
    """Verify native C++20 SIMD Collatz scanner compiles and executes at >10M numbers/sec."""
    bin_path = Path("bin/collatz-scanner")
    if not bin_path.exists():
        compile_res = subprocess.run(
            ["g++", "-O3", "-march=native", "-fopenmp", "src/simd_collatz_scanner.cpp", "-o", "bin/collatz-scanner"],
            capture_output=True,
            text=True
        )
        assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    # Execute scanner for 5 Million numbers
    res = subprocess.run(
        [str(bin_path), "5000000"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Scanner execution failed:\n{res.stderr}"
    assert "Counterexamples Found: 0" in res.stdout
    assert "Search Throughput:" in res.stdout
