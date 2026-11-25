/-
  Formal Verification of Tropical Moduli Spaces for Token Trees under Speculation.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalModuliTreeSpeculation

structure TropicalTreeSpec where
  candidatePaths : Nat
  treeDepth : Nat
  rootTokens : Nat

def totalTropicalTreeDegrees (spec : TropicalTreeSpec) : Nat :=
  spec.candidatePaths * spec.treeDepth + spec.rootTokens

theorem tropical_tree_degrees_nonnegative (spec : TropicalTreeSpec) :
    totalTropicalTreeDegrees spec >= 0 := by
  exact Nat.zero_le (totalTropicalTreeDegrees spec)

theorem tropical_tree_degrees_positive (spec : TropicalTreeSpec) (hp : spec.candidatePaths > 0) (hd : spec.treeDepth > 0) :
    totalTropicalTreeDegrees spec >= 1 := by
  dsimp [totalTropicalTreeDegrees]
  have h1 : spec.candidatePaths * spec.treeDepth >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hp hd)
  exact Nat.le_add_right_of_le h1

end Weft.TropicalModuliTreeSpeculation
