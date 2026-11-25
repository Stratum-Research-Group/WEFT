-- MotivicSectionConjectureInvariants.lean
-- Milestone Paper 275: Asymmetric Motivic Étale Section Conjectures
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure EtaleSectionObstruction where
  dim : Nat
  diffSq : Nat
  normSq : Nat
  thresholdScaled : Nat
  deriving Repr

def isOODAnomaly (o : EtaleSectionObstruction) : Prop :=
  o.diffSq > o.thresholdScaled * o.normSq

theorem obstruction_zero_implies_in_distribution (o : EtaleSectionObstruction) (h : o.diffSq = 0) :
    ¬ (isOODAnomaly o) := by
  intro h_anom
  dsimp [isOODAnomaly] at h_anom
  rw [h] at h_anom
  exact Nat.not_lt_zero (o.thresholdScaled * o.normSq) h_anom

theorem obstruction_monotonicity (d1 d2 thresh norm : Nat) (h_le : d1 ≤ d2) (h_anom : d1 > thresh * norm) :
    d2 > thresh * norm := by
  exact Nat.lt_of_lt_of_le h_anom h_le
