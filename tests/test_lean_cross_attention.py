import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
import subprocess
from pathlib import Path


def test_lean_cross_attention_bounds():
    lean_file = Path(__file__).parent.parent / "formal" / "CrossAttentionBounds.lean"
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(res.stdout.strip()) == 0, f"Unexpected stdout: {res.stdout}"


def test_cross_attention_gqa_reduction_simulation():
    d_model = 8192
    num_heads_q = 64
    num_heads_kv = 8  # Llama-3.3-70B Grouped-Query Attention (GQA)

    d_q = d_model * (num_heads_q * 128)
    d_kv = d_model * (num_heads_kv * 128)

    ratio = d_q / d_kv
    assert ratio == 8.0, f"Expected 8x GQA parameter reduction, got {ratio}x"
