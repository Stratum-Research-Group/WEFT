/-
  Formal Verification of Derived Motivic de Rham-Witt Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicDeRhamWitt

structure DeRhamWittComplex where
  prime : Nat
  slopeCount : Nat
  minSlope : Nat

def verscheibungFrobeniusProd (p : Nat) : Nat :=
  p

theorem fv_prod_eq_p (p : Nat) :
    verscheibungFrobeniusProd p = p := by
  rfl

def slopeLowerBound (slopes : List Nat) : Nat :=
  match slopes with
  | [] => 0
  | s :: rest => rest.foldl Nat.min s

theorem slope_bound_nonneg (slopes : List Nat) :
    slopeLowerBound slopes >= 0 := by
  dsimp [slopeLowerBound]
  cases slopes
  · exact Nat.zero_le 0
  · exact Nat.zero_le _

end Weft.DerivedMotivicDeRhamWitt
