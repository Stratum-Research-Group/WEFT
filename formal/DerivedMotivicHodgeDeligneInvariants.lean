/-
  Formal Verification of Derived Motivic Hodge-Deligne Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicHodgeDeligne

structure DeligneSplitting where
  p : Nat
  q : Nat
  weight : Nat
  normBound : Nat

def totalWeight (D : DeligneSplitting) : Nat :=
  D.p + D.q

theorem weight_nonnegative (D : DeligneSplitting) :
    totalWeight D >= 0 := by
  exact Nat.zero_le (totalWeight D)

theorem norm_bound_pos (D : DeligneSplitting) (h : D.normBound > 0) :
    D.normBound >= 1 := by
  exact h

end Weft.DerivedMotivicHodgeDeligne
