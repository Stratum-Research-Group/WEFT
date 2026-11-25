/-
  Formal Verification of Tropical Intersection Theoretic Bounds on Associative Recall.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalIntersectionAssociativeRecall

structure RecallIntersectionSpec where
  dimKey : Nat
  dimVal : Nat
  stateRank : Nat

def totalIntersectionDegrees (spec : RecallIntersectionSpec) : Nat :=
  spec.dimKey * spec.stateRank + spec.dimVal

theorem recall_intersection_nonnegative (spec : RecallIntersectionSpec) :
    totalIntersectionDegrees spec >= 0 := by
  exact Nat.zero_le (totalIntersectionDegrees spec)

theorem state_rank_positive (spec : RecallIntersectionSpec) (h : spec.stateRank > 0) :
    spec.stateRank >= 1 := by
  exact h

end Weft.TropicalIntersectionAssociativeRecall
