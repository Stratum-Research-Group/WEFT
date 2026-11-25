/-
  Formal Verification of Tropical Toric Speculative Compactification Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalToricCompactificationsSpeculative

structure SpeculativeCone where
  dimension : Nat
  rayCount : Nat
  strataCount : Nat

def maxPathHorizon (C : SpeculativeCone) : Nat :=
  C.dimension + C.rayCount

theorem horizon_ge_dimension (C : SpeculativeCone) :
    maxPathHorizon C >= C.dimension := by
  dsimp [maxPathHorizon]
  exact Nat.le_add_right C.dimension C.rayCount

theorem strata_nonnegative (C : SpeculativeCone) :
    C.strataCount >= 0 := by
  exact Nat.zero_le C.strataCount

end Weft.TropicalToricCompactificationsSpeculative
