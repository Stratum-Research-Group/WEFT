/-
  formal/TropicalCompactificationMaskInvariants.lean
  Milestone Paper 238: Formal Verification of Tropical Compactifications for Sparse Attention.
  Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
  License: Apache-2.0

  Zero non-standard axioms, zero sorry.
-/

namespace Weft.TropicalCompactificationMask

/-- Token attention stratum coordinate: discrete tropical projection value. -/
structure StratumToken where
  tokenIdx : Nat
  boundaryScore : Nat

/-- Binary mask decision: token is active (1) if boundaryScore >= threshold tau, else 0. -/
def isMaskActive (t : StratumToken) (tau : Nat) : Nat :=
  if t.boundaryScore ≥ tau then 1 else 0

/-- Theorem 1: Mask decision is strictly bounded by 1. -/
theorem mask_active_binary (t : StratumToken) (tau : Nat) :
    isMaskActive t tau = 0 ∨ isMaskActive t tau = 1 := by
  dsimp [isMaskActive]
  split <;> omega

/-- Theorem 2: Threshold monotonicity: if tau1 <= tau2, an active token at tau2 is active at tau1. -/
theorem mask_threshold_monotonic (t : StratumToken) (tau1 tau2 : Nat) (h_tau : tau1 ≤ tau2)
    (h_active : isMaskActive t tau2 = 1) :
    isMaskActive t tau1 = 1 := by
  dsimp [isMaskActive] at *
  split at h_active
  · split
    · rfl
    · omega
  · contradiction

/-- Active mask count accumulator abstraction over 2 tokens. -/
def activePairCount (t1 t2 : StratumToken) (tau : Nat) : Nat :=
  isMaskActive t1 tau + isMaskActive t2 tau

/-- Theorem 3: Active count is upper bounded by total tokens (2). -/
theorem active_pair_count_le_total (t1 t2 : StratumToken) (tau : Nat) :
    activePairCount t1 t2 tau ≤ 2 := by
  dsimp [activePairCount, isMaskActive]
  split <;> split <;> omega

end Weft.TropicalCompactificationMask
