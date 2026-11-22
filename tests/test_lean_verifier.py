import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
"""
Tests for Lean 4 Formal Verification & Recursive Self-Improvement Engine.
"""

import pytest
from weft.lean_verifier import Lean4Verifier, LeanSelfCorrectionLoop

def test_lean4_verifier_installed():
    verifier = Lean4Verifier()
    code = "def f (x : Nat) : Nat := x + 1\n#check f\n"
    res = verifier.verify_code(code)
    assert res["success"] is True
    assert res["exit_code"] == 0

def test_lean4_verifier_catches_error():
    verifier = Lean4Verifier()
    # Invalid proof: false is not true
    code = "theorem bad_thm : (1 = 2) := by rfl\n"
    res = verifier.verify_code(code)
    assert res["success"] is False
    assert len(res["errors"]) > 0

def test_lean4_verify_singlet_uncrossing():
    verifier = Lean4Verifier()
    with open("formal/NetworkStatements.lean") as f:
        code = f.read()
    res = verifier.verify_code(code)
    assert res["success"] is True

def test_lean_self_correction_loop():
    loop = LeanSelfCorrectionLoop()
    preamble = """
def s (p q : Bool) : Int :=
  match p, q with
  | false, true  => 1
  | true, false => -1
  | _, _         => 0

def I1 (x y u v : Bool) : Int :=
  if x == u && y == v then 1 else 0

def X1 (x y u v : Bool) : Int :=
  if x == v && y == u then 1 else 0
"""
    stmt = "(x y u v : Bool) : X1 x y u v = I1 x y u v - s x y * s u v"
    # Provide flawed initial tactic 'by rfl' which fails because cases are needed
    flawed_initial = "by rfl"
    
    success, final_code, history = loop.prove_and_refine(
        theorem_name="singlet_uncrossing_auto",
        theorem_stmt=stmt,
        initial_proof=flawed_initial,
        preamble=preamble
    )
    assert success is True
    assert len(history) >= 2 # Proves that iteration 1 failed and iteration 2 auto-repaired
    assert history[0]["success"] is False
    assert history[1]["success"] is True


def test_verify_theorem_strict_with_axiom_detection():
    verifier = Lean4Verifier()
    # 1. Clean theorem with zero axioms
    clean_code = """
theorem clean_identity (x : Nat) : x + 0 = x := by
  rfl
"""
    clean_res = verifier.verify_theorem_strict("clean_identity", clean_code)
    assert clean_res["verified"] is True
    assert clean_res["axioms"] == []

    # 2. Theorem relying on sorryAx
    sorry_code = """
theorem sorry_identity (x : Nat) : x + 0 = x := by
  sorry
"""
    sorry_res = verifier.verify_theorem_strict("sorry_identity", sorry_code)
    assert sorry_res["verified"] is False
    assert "sorry" in sorry_res["reason"].lower() or "sorryAx" in sorry_res["axioms"]

    # 3. Theorem relying on custom unproven axiom
    cheating_code = """
axiom cheat_ax (x : Nat) : x + 0 = x

theorem cheat_thm (x : Nat) : x + 0 = x := by
  exact cheat_ax x
"""
    cheat_res = verifier.verify_theorem_strict("cheat_thm", cheating_code)
    assert cheat_res["verified"] is False
    assert "cheat_ax" in cheat_res["axioms"]
    assert "unauthorized axioms" in cheat_res["reason"].lower()

