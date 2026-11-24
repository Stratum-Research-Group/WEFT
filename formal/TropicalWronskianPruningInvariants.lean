-- TropicalWronskianPruningInvariants.lean
-- Milestone Paper 253: Asymmetric Tropical Wronskian Determinants & Metric Graph Linear Series
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure TropicalHead where
  headId : Nat
  degree : Nat
  slopeValuation : Nat
  deriving Repr, DecidableEq

def tropicalWronskianValuation (h1 h2 : TropicalHead) : Nat :=
  if h1.slopeValuation >= h2.slopeValuation then
    h1.slopeValuation - h2.slopeValuation
  else
    h2.slopeValuation - h1.slopeValuation

theorem tropical_wronskian_nonneg (h1 h2 : TropicalHead) :
    tropicalWronskianValuation h1 h2 >= 0 := by
  dsimp [tropicalWronskianValuation]
  split <;> omega

theorem tropical_wronskian_self (h : TropicalHead) :
    tropicalWronskianValuation h h = 0 := by
  dsimp [tropicalWronskianValuation]
  split <;> omega

def isTropicallyRedundant (h1 h2 : TropicalHead) (threshold : Nat) : Bool :=
  tropicalWronskianValuation h1 h2 <= threshold

theorem redundancy_symmetry (h1 h2 : TropicalHead) (threshold : Nat) :
    isTropicallyRedundant h1 h2 threshold = isTropicallyRedundant h2 h1 threshold := by
  dsimp [isTropicallyRedundant, tropicalWronskianValuation]
  by_cases h : h1.slopeValuation >= h2.slopeValuation
  · by_cases h_eq : h1.slopeValuation = h2.slopeValuation
    · simp [h_eq]
    · have h_lt : ¬ (h2.slopeValuation >= h1.slopeValuation) := by omega
      simp [h, h_lt]
  · have h_gt : h2.slopeValuation >= h1.slopeValuation := by omega
    simp [h, h_gt]
