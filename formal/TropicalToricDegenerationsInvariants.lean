/-
  Formal Verification of Tropical Toric Degenerations Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalToricDegenerations

structure TropicalDegeneration where
  dimManifold : Nat
  numFacets : Nat
  fanLatticeRank : Nat
  h_dim : dimManifold <= fanLatticeRank

theorem tropical_dim_preserved (T : TropicalDegeneration) :
    T.dimManifold <= T.fanLatticeRank :=
  T.h_dim

def maxApproximationError (maxStep : Nat) (numSlices : Nat) : Nat :=
  maxStep * numSlices

theorem max_error_additive (s : Nat) (n1 n2 : Nat) :
    maxApproximationError s (n1 + n2) = maxApproximationError s n1 + maxApproximationError s n2 := by
  dsimp [maxApproximationError]
  exact Nat.mul_add s n1 n2

end Weft.TropicalToricDegenerations
