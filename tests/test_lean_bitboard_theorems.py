import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
import pytest
from weft.lean_verifier import Lean4Verifier

def test_lean4_verify_bitboard_theorems():
    verifier = Lean4Verifier()
    with open("formal/BitboardTheorems.lean") as f:
        code = f.read()
    res = verifier.verify_code(code)
    assert res["success"] is True
    assert res["exit_code"] == 0
