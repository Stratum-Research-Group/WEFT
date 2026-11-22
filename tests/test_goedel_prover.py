import os
from pathlib import Path
import pytest
import torch
from weft.goedel_prover import GoedelSubBitProver

def test_goedel_subbit_prover():
    models_root = Path(os.environ.get("WEFT_MODELS_ROOT", "/run/media/akarta/AI/models"))
    shard = models_root / "goedel-prover-32b-subbit" / "model-00001-of-00014_subbit.weft"
    if not shard.exists():
        pytest.skip(f"Shard file {shard} does not exist.")
    prover = GoedelSubBitProver([str(shard)])
    
    # 1. Test sub-bit scoring layer
    dummy_tokens = torch.tensor([[10, 20, 30]])
    logits = prover.evaluate_tactic_logits(dummy_tokens)
    assert logits.shape[0] == 1
    assert not torch.isnan(logits).any()

    # 2. Test theorem proving loop
    preamble = "def double (n : Nat) : Nat := n + n"
    thm_name = "double_zero"
    stmt = ": double 0 = 0"
    
    result = prover.synthesize_and_prove(
        preamble=preamble,
        theorem_name=thm_name,
        statement=stmt,
        candidate_tactics=["by rfl", "by omega"]
    )
    assert result["success"] is True
    assert result["verified_tactic"] in ["by rfl", "by omega"]
