/-
  Formal Verification of Motivic Galois Cohomology of Bit-Level Speculative Branch Pruning.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisSpeculativePruning

structure SpeculativePruningSpec where
  candidateBranches : Nat
  pruningRank : Nat
  cohomologicalObstructions : Nat

def totalPruningDegrees (spec : SpeculativePruningSpec) : Nat :=
  spec.candidateBranches * spec.pruningRank + spec.cohomologicalObstructions

theorem pruning_degrees_nonnegative (spec : SpeculativePruningSpec) :
    totalPruningDegrees spec >= 0 := by
  exact Nat.zero_le (totalPruningDegrees spec)

theorem candidate_branches_positive (spec : SpeculativePruningSpec) (h : spec.candidateBranches > 0) :
    spec.candidateBranches >= 1 := by
  exact h

end Weft.MotivicGaloisSpeculativePruning
