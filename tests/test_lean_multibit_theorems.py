"""
Unit tests for Lean 4 Multi-Bit Continuum Invariant Theorems.
"""

import os
import subprocess
import pytest


def test_lean4_multibit_theorems_kernel_verification():
    lean_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "formal", "MultiBitTheorems.lean")
    assert os.path.exists(lean_file), f"MultiBitTheorems.lean not found at {lean_file}"

    result = subprocess.run(
        ["lean", lean_file],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Lean 4 kernel verification failed:\n{result.stderr}\n{result.stdout}"
    assert result.stdout.strip() == "", f"Lean 4 kernel emitted unexpected warnings/errors: {result.stdout}"
