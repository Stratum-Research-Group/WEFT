import pytest
from weft.lean_verifier import Lean4Verifier

def test_lean4_verify_multimodal_theorems():
    verifier = Lean4Verifier()
    with open("formal/MultimodalTheorems.lean") as f:
        code = f.read()
    res = verifier.verify_code(code)
    assert res["success"] is True
    assert res["exit_code"] == 0
