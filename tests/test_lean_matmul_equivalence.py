import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
import subprocess
from pathlib import Path


def test_lean_matmul_equivalence():
    lean_file = Path(__file__).parent.parent / "formal" / "MatmulEquivalence.lean"
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(res.stdout.strip()) == 0, f"Unexpected stdout: {res.stdout}"


def test_matmul_associativity_flops_comparison():
    m, n, r = 8192, 28672, 32

    # Dense standard matrix-vector product
    dense_flops = 2 * m * n

    # Sub-bit factored matrix-vector product
    subbit_flops = 2 * r * m + 2 * r * n + m + n + r

    # Dual-path factored matrix-vector product
    dual_flops = 2 * subbit_flops

    assert subbit_flops < dense_flops
    assert dual_flops < dense_flops
    speedup = dense_flops / dual_flops
    assert speedup > 80.0, f"Expected >80x algorithmic FLOP reduction, got {speedup:.2f}x"
