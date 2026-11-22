import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
import subprocess
from pathlib import Path


def test_lean_orthonormality_bounds():
    lean_file = Path(__file__).parent.parent / "formal" / "OrthonormalityBounds.lean"
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(res.stdout.strip()) == 0, f"Unexpected stdout: {res.stdout}"


def test_hypercube_energy_invariants():
    import torch

    d = 1024
    # Simulate binary hypercube coordinate vector {-1, +1}^d
    b = torch.randint(0, 2, (d,), dtype=torch.float32) * 2 - 1
    # Check that ||b||^2 == d exactly
    norm_sq = torch.sum(b * b).item()
    assert abs(norm_sq - d) < 1e-4, f"Expected hypercube norm squared {d}, got {norm_sq}"
