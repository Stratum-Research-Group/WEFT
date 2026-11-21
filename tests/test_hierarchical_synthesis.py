"""Unit tests for Hierarchical Program Synthesis & Multi-File Engine."""
import pytest
from weft.hierarchical_synthesis_engine import HierarchicalSynthesisEngine

def test_multi_file_syntax_and_lean_contract():
    engine = HierarchicalSynthesisEngine()

    files = {
        "src/core.py": """
def compute_hash(data: str) -> int:
    return len(data) * 31
""",
        "src/service.py": """
from src.core import compute_hash

class AuthService:
    def verify(self, token: str) -> bool:
        return compute_hash(token) > 0
"""
    }

    # Formal Lean 4 contract representing interface invariants
    contracts = [
        """
        theorem hash_contract (n : Nat) (h : n > 0) : n * 31 > 0 := by
          omega
        """
    ]

    res = engine.synthesize_multi_file_project(files, contracts)
    assert res["all_syntax_valid"] is True
    assert res["all_contracts_verified"] is True
    assert res["file_status"]["src/core.py"] is True
    assert res["file_status"]["src/service.py"] is True
