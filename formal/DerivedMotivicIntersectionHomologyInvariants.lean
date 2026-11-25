/-
  Formal Verification of Derived Motivic Intersection Homology Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicIntersectionHomology

structure StratifiedSpace where
  dim : Nat
  strataCount : Nat

def middlePerversity (k : Nat) : Nat :=
  if k >= 2 then (k - 2) / 2 else 0

theorem middle_perversity_le (k : Nat) :
    middlePerversity k <= k := by
  dsimp [middlePerversity]
  split
  · exact Nat.le_trans (Nat.div_le_self (k - 2) 2) (Nat.sub_le k 2)
  · exact Nat.zero_le k

end Weft.DerivedMotivicIntersectionHomology
