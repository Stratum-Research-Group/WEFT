import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
import subprocess
from pathlib import Path


def test_lean_dual_path_alignment():
    lean_file = Path(__file__).parent.parent / "formal" / "DualPathAlignment.lean"
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(res.stdout.strip()) == 0, f"Unexpected stdout: {res.stdout}"


def test_dual_path_ternary_state_simulation():
    import torch

    # Simulate two binary coordinate paths {-1, +1}
    N = 1000
    p1 = torch.randint(0, 2, (N,), dtype=torch.float32) * 2 - 1
    p2 = torch.randint(0, 2, (N,), dtype=torch.float32) * 2 - 1

    # Superposition
    combo = p1 + p2
    unique_vals = sorted(list(torch.unique(combo).tolist()))

    # Possible values of {-1, 1} + {-1, 1} are {-2.0, 0.0, 2.0}
    assert unique_vals == [-2.0, 0.0, 2.0], f"Expected ternary state values [-2, 0, 2], got {unique_vals}"
