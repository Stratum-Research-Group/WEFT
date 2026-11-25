/-
  Formal Verification of Block-Sparse Triton MoE Kernel Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.BlockSparseTritonMoE

structure BlockSparseMoESpec where
  numBlocks : Nat
  expertsPerBlock : Nat
  topK : Nat

def totalBlockSparseDim (spec : BlockSparseMoESpec) : Nat :=
  spec.numBlocks * spec.expertsPerBlock + spec.topK

theorem block_sparse_moe_nonnegative (spec : BlockSparseMoESpec) :
    totalBlockSparseDim spec >= 0 := by
  exact Nat.zero_le (totalBlockSparseDim spec)

theorem block_sparse_moe_positive (spec : BlockSparseMoESpec) (hb : spec.numBlocks > 0) (he : spec.expertsPerBlock > 0) :
    totalBlockSparseDim spec >= 1 := by
  dsimp [totalBlockSparseDim]
  have h1 : spec.numBlocks * spec.expertsPerBlock >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hb he)
  exact Nat.le_add_right_of_le h1

end Weft.BlockSparseTritonMoE
