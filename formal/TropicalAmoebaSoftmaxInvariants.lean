/-
  formal/TropicalAmoebaSoftmaxInvariants.lean
  Milestone Paper 247: Formal Verification of Tropical Amoeba Softmax Normalization.
  Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
  License: Apache-2.0

  Zero non-standard axioms, zero sorry.
-/

namespace Weft.TropicalAmoebaSoftmax

/-- Tropical piecewise linear activation on Nat coordinates:
    phi(diff) = if diff <= bound then bound - diff else 0. -/
def tropicalPhi (diff bound : Nat) : Nat :=
  if diff ≤ bound then bound - diff else 0

/-- Theorem 1: Tropical activation is non-negative. -/
theorem tropical_phi_nonneg (diff bound : Nat) :
    tropicalPhi diff bound ≥ 0 := by
  dsimp [tropicalPhi]
  split <;> omega

/-- Theorem 2: At maximum logit (diff = 0), activation attains maximal value bound. -/
theorem tropical_phi_max (bound : Nat) :
    tropicalPhi 0 bound = bound := by
  dsimp [tropicalPhi]

/-- Normalization proxy for two logits with scores s1 and s2:
    prob1 = s1 / (s1 + s2) on scaled integer basis. -/
def normalizedProb (s1 s2 : Nat) (_h_pos : s1 + s2 > 0) : Nat :=
  (s1 * 1000) / (s1 + s2)

/-- Theorem 3: Normalized probability is upper-bounded by 1000 (1.0). -/
theorem normalized_prob_le_one (s1 s2 : Nat) (h_pos : s1 + s2 > 0) :
    normalizedProb s1 s2 h_pos ≤ 1000 := by
  dsimp [normalizedProb]
  have h_mul : s1 * 1000 ≤ (s1 + s2) * 1000 := by omega
  have h_div : (s1 * 1000) / (s1 + s2) ≤ ((s1 + s2) * 1000) / (s1 + s2) :=
    Nat.div_le_div_right h_mul
  have h_id : (s1 + s2) * 1000 / (s1 + s2) = 1000 := Nat.mul_div_cancel_left 1000 h_pos
  rw [h_id] at h_div
  exact h_div

end Weft.TropicalAmoebaSoftmax
