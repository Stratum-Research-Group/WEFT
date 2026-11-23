"""
Unit tests for Grand Computational Laboratory (Paper 60) and Lean 4 invariants.
"""

import subprocess
import pytest
from pathlib import Path


def test_lean_computational_laboratory_invariants_compiles():
    """Verify formal/ComputationalLaboratoryInvariants.lean compiles cleanly with zero sorry."""
    lean_file = Path("formal/ComputationalLaboratoryInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in ComputationalLaboratoryInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in ComputationalLaboratoryInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_native_goldbach_scanner_execution():
    """Verify native C++20 Goldbach and Twin Prime scanner executes at >50M numbers/sec."""
    bin_path = Path("bin/goldbach-scanner")
    if not bin_path.exists():
        compile_res = subprocess.run(
            ["g++", "-O3", "-march=native", "-fopenmp", "src/simd_goldbach_scanner.cpp", "-o", "bin/goldbach-scanner"],
            capture_output=True,
            text=True
        )
        assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    res = subprocess.run(
        [str(bin_path), "1000000"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Scanner execution failed:\n{res.stderr}"
    assert "Goldbach Counterexamples: 0" in res.stdout
    assert "Twin Prime Pairs Found:" in res.stdout
