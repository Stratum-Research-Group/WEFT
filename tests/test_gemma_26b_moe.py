"""
Unit tests for Gemma-4 26B-A4B-it Sparse MoE Sub-Bit Deconstruction and Lean 4 invariants.
"""

import subprocess
import pytest
import json
from pathlib import Path


def test_lean_gemma_26b_moe_invariants_compiles():
    """Verify formal/Gemma26BMoEInvariants.lean compiles cleanly with zero sorry."""
    lean_file = Path("formal/Gemma26BMoEInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in Gemma26BMoEInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in Gemma26BMoEInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_gemma_26b_subbit_artifacts_exist():
    """Verify gemma-4-26B subbit artifacts are present on AI drive."""
    cert_path = Path("/run/media/akarta/AI/models/gemma-4-26B-A4B-it-subbit_certificate.json")
    weft_path = Path("/run/media/akarta/AI/models/gemma-4-26B-A4B-it-subbit.weft")
    gguf_path = Path("/run/media/akarta/AI/models/gemma-4-26B-A4B-it-subbit.gguf")

    assert cert_path.exists()
    assert weft_path.exists()
    assert gguf_path.exists()

    cert = json.loads(cert_path.read_text())
    assert cert["num_experts"] == 128
    assert cert["active_experts_per_token"] == 8
    assert cert["bpw"] == 0.0382
    assert cert["verified_in_lean"] is True
