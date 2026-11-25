/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

/-!
# Sub-Bit Vocabulary Un-Embedding and Logit Projection Invariants

This module formalizes mathematical invariance guarantees for sub-bit un-embedding heads
mapping intermediate sub-bit representations into discrete token selections.
-/

namespace Weft.VocabularyProjection

/-- Theorem: Product of two bipolar bits in {-1, 1} is always in {-1, 1} -/
theorem bipolar_prod_mem (a b : Int) (ha : a = 1 ∨ a = -1) (hb : b = 1 ∨ b = -1) :
    a * b = 1 ∨ a * b = -1 := by
  cases ha with
  | inl ha1 =>
    cases hb with
    | inl hb1 => rw [ha1, hb1]; decide
    | inr hb2 => rw [ha1, hb2]; decide
  | inr ha2 =>
    cases hb with
    | inl hb1 => rw [ha2, hb1]; decide
    | inr hb2 => rw [ha2, hb2]; decide

/-- Lemma: Invariance of strict order under multiplication by positive integer -/
theorem mul_pos_strict_mono (l₁ l₂ : Int) (scale : Int) (h_scale : scale > 0) :
    l₁ < l₂ → l₁ * scale < l₂ * scale := by
  intro h
  exact Int.mul_lt_mul_of_pos_right h h_scale

/-- Invariance: Weak inequality preserved under positive scalar multiplication -/
theorem mul_pos_le_mono (l₁ l₂ : Int) (scale : Int) (h_scale : scale > 0) :
    l₁ ≤ l₂ → l₁ * scale ≤ l₂ * scale := by
  intro h
  have h_nonneg : 0 ≤ scale := Int.le_of_lt h_scale
  exact Int.mul_le_mul_of_nonneg_right h h_nonneg

/-- Invariance of argmax choice between two candidate logits under positive scaling -/
theorem greedy_bipolar_token_choice (l₁ l₂ : Int) (scale : Int) (h_scale : scale > 0) :
    (l₁ ≥ l₂) → (l₁ * scale ≥ l₂ * scale) := by
  intro h
  have h_le : l₂ ≤ l₁ := h
  have h_scaled := mul_pos_le_mono l₂ l₁ scale h_scale h_le
  exact h_scaled

end Weft.VocabularyProjection
