-- TropicalFlagVarietiesInvariants.lean
-- Milestone Paper 289: Asymmetric Tropical Toric Degenerations of Flag Varieties
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure FlagRankState where
  fullDim : Nat
  reducedRank : Nat
  singularValueTailScaled : Nat
  toleranceScaled : Nat
  deriving Repr

def isRankApproximationValid (s : FlagRankState) : Prop :=
  s.singularValueTailScaled ≤ s.toleranceScaled

theorem rank_approximation_guaranteed (s : FlagRankState) (h_tail : s.singularValueTailScaled = 0) :
    isRankApproximationValid s := by
  dsimp [isRankApproximationValid]
  rw [h_tail]
  exact Nat.zero_le s.toleranceScaled

theorem tail_bound_monotonicity (t1 t2 tol : Nat) (h_le : t1 ≤ t2) (h_valid : t2 ≤ tol) :
    t1 ≤ tol := by
  exact Nat.le_trans h_le h_valid
