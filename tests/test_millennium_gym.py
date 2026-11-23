"""
Unit and property-based test suite for Millennium Prize Mathematical Gym
and Lean 4 verified formal invariants.
"""

import subprocess
import pytest
from pathlib import Path
from weft.millennium_gym import MillenniumMathGym, MILLENNIUM_PROBLEMS


def test_lean_millennium_gym_invariants_compiles():
    """Verify formal/MillenniumGymInvariants.lean compiles with zero axioms and zero sorry."""
    lean_file = Path("formal/MillenniumGymInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in MillenniumGymInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in MillenniumGymInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_millennium_gym_initialization():
    """Verify gym initialization and problem catalog completeness."""
    gym = MillenniumMathGym()
    assert len(gym.problems) == 7
    ids = [p.problem_id for p in gym.problems]
    assert ids == [f"MPP-0{i}" for i in range(1, 8)]


def test_millennium_gym_full_benchmark_run():
    """Execute complete gym benchmark run across all 7 problems."""
    gym = MillenniumMathGym()
    results = gym.run_benchmark()
    assert len(results) == 7

    for r in results:
        assert r.proof_tree_valid is True
        assert r.lean_invariant_verified is True
        assert r.algebraic_reduction_steps >= 9
        assert r.eval_latency_ms < 100.0
