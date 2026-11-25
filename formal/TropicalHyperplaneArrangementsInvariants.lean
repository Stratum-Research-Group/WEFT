/-
  Formal Verification of Tropical Hyperplane Arrangements Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalHyperplanes

structure HyperplaneArrangement where
  numHyperplanes : Nat
  ambientDim : Nat
  rank : Nat
  h_rank_le : rank <= ambientDim
  h_rank_m : rank <= numHyperplanes

theorem rank_le_ambient (A : HyperplaneArrangement) :
    A.rank <= A.ambientDim :=
  A.h_rank_le

def maximumChambers (n : Nat) (d : Nat) : Nat :=
  n * d + 1

theorem maximum_chambers_nonneg (n d : Nat) :
    maximumChambers n d >= 1 := by
  dsimp [maximumChambers]
  exact Nat.le_add_left 1 (n * d)

end Weft.TropicalHyperplanes
