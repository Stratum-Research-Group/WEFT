import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
"""
Unit tests for Lean 4 Verified Zipf-Optimal Vocabulary Partitioning Theorems.
"""
import pytest
import os
import subprocess

def test_lean4_zipf_theorems_kernel_verification():
    lean_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "formal", "ZipfTheorems.lean")
    assert os.path.exists(lean_file), f"ZipfTheorems.lean not found at {lean_file}"
    
    # Run through the official Lean 4 kernel
    result = subprocess.run(
        ["lean", lean_file],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Lean 4 kernel verification failed:\n{result.stderr}\n{result.stdout}"
    assert result.stdout.strip() == "", f"Lean 4 kernel emitted unexpected warnings/errors: {result.stdout}"
