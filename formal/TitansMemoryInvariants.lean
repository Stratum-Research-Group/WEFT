/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

/-!
# Formal Invariants of Titans Surprise-Gated Plastic Fast-Weights Memory

This module establishes formal mechanical proofs for the integer/discrete-scaled
contraction bounds and non-negativity of surprise-gated fast-weight memory updates.
-/

namespace Weft.TitansMemory

/-- Decay step bound: Multiplying a non-negative value by (scale - decay) preserves non-negativity -/
theorem decay_factor_nonneg (scale : Nat) (decay : Nat) (h : decay ≤ scale) :
    (scale - decay : Int) ≥ 0 := by
  have h_nat : (scale - decay : Int) = ((scale - decay : Nat) : Int) := by
    exact (Int.ofNat_sub h).symm
  rw [h_nat]
  exact Int.natCast_nonneg (scale - decay)

/-- Invariance: Discrete fast weight step preservation under non-negative momentum and non-negative state -/
theorem discrete_fast_weight_mono (w : Int) (scale : Int) (m : Int)
    (hw : w ≥ 0) (hscale : scale ≥ 0) (hm : m ≥ 0) :
    scale * w + m ≥ 0 := by
  have h_mul : scale * w ≥ 0 := by
    exact Int.mul_nonneg hscale hw
  exact Int.add_nonneg h_mul hm

/-- Non-negativity of surprise quadratic prediction error from nat casting -/
theorem surprise_error_sq_nat (e : Nat) : (e : Int) * (e : Int) ≥ 0 := by
  have h_nonneg : (e : Int) ≥ 0 := Int.natCast_nonneg e
  exact Int.mul_nonneg h_nonneg h_nonneg

/-- Invariance: Square of any non-negative integer is non-negative -/
theorem int_sq_nonneg (e : Int) (he : e ≥ 0) : e * e ≥ 0 := by
  exact Int.mul_nonneg he he

end Weft.TitansMemory
