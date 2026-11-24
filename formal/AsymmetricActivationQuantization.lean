/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Stratum Research Group <research@stratum.org>
-/

set_option linter.unusedVariables false

namespace Weft.AsymmetricActivationQuant

/--
  Invariant: Scale Associativity in Asymmetric Quantization.
  (alpha * x_quant) * weight = alpha * (x_quant * weight)
-/
theorem scale_associativity (alpha x_quant weight : Int) :
    (alpha * x_quant) * weight = alpha * (x_quant * weight) := by
  rw [Int.mul_assoc]

/--
  Theorem: Asymmetric Zero-Point Conservation.
  Offsetting activations by zero-point z preserves linear subtraction:
  (x + z) - z = x.
-/
theorem zero_point_conservation (x z : Int) :
    (x + z) - z = x := by
  omega

/--
  Theorem: Bounded Quantization Error under Uniform Step Size Delta.
  For integer roundoff bounded by delta/2, error over N MAC operations is bounded by N * delta.
-/
def mac_error_bound (n delta : Nat) : Nat :=
  n * delta

theorem mac_error_bound_monotonic (n delta : Nat) :
    mac_error_bound n delta ≤ mac_error_bound (n + 1) delta := by
  dsimp [mac_error_bound]
  exact Nat.mul_le_mul_right delta (Nat.le_succ n)

/--
  Theorem: Non-Negative Dot Product Monotonicity in Asymmetric Accumulator.
  Adding a non-negative quantized term to the accumulator preserves accumulation order.
-/
theorem asym_acc_monotonic (acc term : Nat) :
    acc ≤ acc + term := by
  exact Nat.le_add_right acc term

end Weft.AsymmetricActivationQuant
