"""
Unit tests for Conjecture Discovery Gym (2,000 Theorems) and Lean 4 invariants.
"""

import subprocess
import pytest
from pathlib import Path
from weft.discovery_gym import ConjectureDiscoveryGym


def test_lean_conjecture_discovery_invariants_compiles():
    """Verify formal/ConjectureDiscoveryInvariants.lean compiles with zero sorry and zero axioms."""
    lean_file = Path("formal/ConjectureDiscoveryInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in ConjectureDiscoveryInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in ConjectureDiscoveryInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_conjecture_discovery_gym_full_scale_2000():
    """Verify discovery gym generates and resolves exactly 2,000 theorems with step-by-step proofs."""
    gym = ConjectureDiscoveryGym()
    results = gym.run_discovery_gym(2000)
    assert len(results) == 2000

    # Validate structure and step completeness
    for idx, thm in enumerate(results, 1):
        assert thm.theorem_id == f"THM-{idx:04d}"
        assert len(thm.derivation_steps) == 5
        assert thm.resolution_status == "VERIFIED_STEPWISE"
        for step in thm.derivation_steps:
            assert step.verified_in_lean is True
            assert len(step.intermediate_state) > 0
            assert len(step.rule_applied) > 0
