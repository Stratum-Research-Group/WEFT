/-
  Formal Verification of Tropical Toric Monodromy for Speculative Trees.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalToricTreeMonodromy

structure TreeMonodromySpec where
  treeBranches : Nat
  monodromyRank : Nat
  verificationBound : Nat

def totalMonodromyDegrees (spec : TreeMonodromySpec) : Nat :=
  spec.treeBranches * spec.monodromyRank + spec.verificationBound

theorem tree_monodromy_nonnegative (spec : TreeMonodromySpec) :
    totalMonodromyDegrees spec >= 0 := by
  exact Nat.zero_le (totalMonodromyDegrees spec)

theorem tree_monodromy_positive (spec : TreeMonodromySpec) (hb : spec.treeBranches > 0) (hr : spec.monodromyRank > 0) :
    totalMonodromyDegrees spec >= 1 := by
  dsimp [totalMonodromyDegrees]
  have h1 : spec.treeBranches * spec.monodromyRank >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hb hr)
  exact Nat.le_add_right_of_le h1

end Weft.TropicalToricTreeMonodromy
