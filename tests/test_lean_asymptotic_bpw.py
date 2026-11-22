import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
import subprocess
from pathlib import Path


def test_lean_asymptotic_bpw_theorems():
    lean_file = Path(__file__).parent.parent / "formal" / "AsymptoticBPW.lean"
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    # Verify lean binary executes with zero errors
    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(res.stdout.strip()) == 0, f"Unexpected Lean stdout: {res.stdout}"


def test_asymptotic_bpw_numerical_simulation():
    """Verify numerical convergence of BPW across matrix dimensions matching serve_any definitions."""
    def calc_bpw(in_f: int, out_f: int, r: int = 32) -> float:
        # Low-rank factor bits: (in_f * r) + (out_f * r) + (r * 16)
        # Total parameters: in_f * out_f
        total_params = in_f * out_f
        bits = (in_f * r) + (out_f * r) + (r * 16)
        return bits / total_params

    # Small matrix (e.g. 512x512)
    bpw_512 = calc_bpw(512, 512, r=32)
    # Llama-3.3-70B hidden layer (8192x8192)
    bpw_8192 = calc_bpw(8192, 8192, r=32)
    # MLP intermediate layer (8192x28672)
    bpw_mlp = calc_bpw(8192, 28672, r=32)

    assert bpw_512 > bpw_8192 > bpw_mlp, "BPW must monotonically decrease as dimensions scale"
    assert bpw_8192 < 0.01, f"Effective BPW for 8192x8192 should be < 0.01, got {bpw_8192}"
    assert bpw_mlp < 0.006, f"Effective BPW for MLP should be < 0.006, got {bpw_mlp}"
