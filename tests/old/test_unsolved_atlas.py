"""
Unit tests for Unsolved Millennium Problems Atlas and Lean 4 invariants.
"""

import subprocess
import pytest
from pathlib import Path
from weft.unsolved_atlas import UnsolvedFrontierAtlas, UNSOLVED_FRONTIERS


def test_lean_unsolved_invariants_compiles():
    """Verify formal/UnsolvedFrontierInvariants.lean compiles with zero sorry and zero axioms."""
    lean_file = Path("formal/UnsolvedFrontierInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in UnsolvedFrontierInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in UnsolvedFrontierInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_unsolved_atlas_structure():
    """Verify exactly 6 active open problems are indexed."""
    atlas = UnsolvedFrontierAtlas()
    assert len(atlas.frontiers) == 6
    for f in atlas.frontiers:
        assert "UNSOLVED" in f.status_as_of_2026
        assert len(f.recent_known_results) >= 3
        assert f.difficulty_score >= 9


def test_unsolved_atlas_audit_execution():
    """Verify atlas audit execution."""
    atlas = UnsolvedFrontierAtlas()
    items = atlas.audit_frontiers()
    assert len(items) == 6
    for it in items:
        assert it["eval_ms"] < 50.0
