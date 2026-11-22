"""Test suite verifying formal Lean 4 BitNet b1.58 Ternary Quantization Bounds."""

import subprocess
import shutil
import pytest
from pathlib import Path

LEAN_FILE = Path(__file__).parent.parent / "formal" / "TernaryQuantizationBounds.lean"


def test_lean_ternary_file_exists():
    assert LEAN_FILE.exists(), f"Lean proof file missing: {LEAN_FILE}"


@pytest.mark.skipif(shutil.which("lean") is None, reason="Lean 4 compiler not installed")
def test_lean_ternary_theorems_compile():
    """Verify that Lean 4 formally compiles and checks all theorems in TernaryQuantizationBounds.lean."""
    res = subprocess.run(
        ["lean", str(LEAN_FILE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0, f"Lean verification failed with error:\n{res.stderr}\n{res.stdout}"


def test_ternary_theorems_present():
    """Verify that all required theorems are written into the proof file."""
    content = LEAN_FILE.read_text(encoding="utf-8")
    expected_theorems = [
        "ternary_zero_input",
        "max_ternary_deviation_le_step",
        "ternary_scaled_energy_zero_elements",
        "ternary_scale_sum_mono",
        "max_ternary_accum_mono",
        "max_ternary_accum_zero",
        "max_ternary_accum_add",
    ]
    for thm in expected_theorems:
        assert f"theorem {thm}" in content, f"Expected theorem '{thm}' not declared in {LEAN_FILE.name}"
