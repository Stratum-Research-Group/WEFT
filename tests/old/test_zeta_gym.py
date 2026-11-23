"""
Unit tests for Riemann Zeta Gym and Lean 4 certified spectral invariants.
"""

import subprocess
import pytest
from pathlib import Path
from weft.zeta_gym import ZetaGymEngine, ZETA_TARGETS


def test_lean_zeta_gym_invariants_compiles():
    """Verify formal/ZetaGymInvariants.lean compiles cleanly with zero sorry and zero axioms."""
    lean_file = Path("formal/ZetaGymInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in ZetaGymInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in ZetaGymInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_zeta_gym_targets_catalog():
    """Verify targets count and schema."""
    gym = ZetaGymEngine()
    assert len(gym.targets) == 4
    for t in gym.targets:
        assert t.target_id.startswith("ZETA-")
        assert len(t.formal_lean_theorem) > 0


def test_zeta_gym_execution():
    """Run full zeta gym benchmarks."""
    gym = ZetaGymEngine()
    reports = gym.run_gym()
    assert len(reports) == 4
    for r in reports:
        assert r.lean_passed is True
        assert r.simulated_zeros_checked >= 100000
        assert r.empirical_accuracy > 99.0
        assert r.eval_latency_ms < 50.0
