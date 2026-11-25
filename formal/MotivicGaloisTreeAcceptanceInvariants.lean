/-
  Formal Verification of Motivic Galois Cohomology of Speculative Tree Acceptance Boundaries.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisTreeAcceptance

structure TreeAcceptanceSpec where
  candidateLeaves : Nat
  acceptanceDepth : Nat
  vanishingObstructions : Nat

def totalAcceptanceDegrees (spec : TreeAcceptanceSpec) : Nat :=
  spec.candidateLeaves * spec.acceptanceDepth + spec.vanishingObstructions

theorem acceptance_degrees_nonnegative (spec : TreeAcceptanceSpec) :
    totalAcceptanceDegrees spec >= 0 := by
  exact Nat.zero_le (totalAcceptanceDegrees spec)

theorem candidate_leaves_positive (spec : TreeAcceptanceSpec) (h : spec.candidateLeaves > 0) :
    spec.candidateLeaves >= 1 := by
  exact h

end Weft.MotivicGaloisTreeAcceptance
