-- TropicalBergmanMoEInvariants.lean
-- Milestone Paper 256: Asymmetric Tropical Hyperplane Arrangements & Bergman Fans for Sparse MoE Expert Selection
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure ExpertValuation where
  expertId : Nat
  valuation : Nat
  deriving Repr, DecidableEq

def tropicalDiff (v1 v2 : Nat) : Nat :=
  if v1 >= v2 then v1 - v2 else v2 - v1

theorem tropical_diff_nonneg (v1 v2 : Nat) : tropicalDiff v1 v2 >= 0 := by
  exact Nat.zero_le _

theorem tropical_diff_symm (v1 v2 : Nat) : tropicalDiff v1 v2 = tropicalDiff v2 v1 := by
  dsimp [tropicalDiff]
  by_cases h : v1 >= v2
  · by_cases h_eq : v1 = v2
    · simp [h_eq]
    · have h_lt : ¬ (v2 >= v1) := by omega
      simp [h, h_lt]
  · have h_gt : v2 >= v1 := by omega
    simp [h, h_gt]

theorem tropical_shift_invariance (v1 v2 c : Nat) :
    tropicalDiff (v1 + c) (v2 + c) = tropicalDiff v1 v2 := by
  dsimp [tropicalDiff]
  split
  · split
    · omega
    · omega
  · split
    · omega
    · omega

def selectChamber (v1 v2 threshold : Nat) : Bool :=
  tropicalDiff v1 v2 <= threshold

theorem select_chamber_symm (v1 v2 threshold : Nat) :
    selectChamber v1 v2 threshold = selectChamber v2 v1 threshold := by
  dsimp [selectChamber]
  rw [tropical_diff_symm v1 v2]
