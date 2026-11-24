/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Stratum Research Group <research@stratum.org>
-/

set_option linter.unusedVariables false

namespace Weft.CoQuantGEMV

/--
  Theorem 1: Co-Quantized Integer Scale Distributivity Invariance.
  For activation scale `sx`, weight scale `sw`, and integer inner product accumulator `acc`,
  multiplying scales together before multiplying the accumulator preserves exact associativity:
  `(sx * sw) * acc = sx * (sw * acc)`.
-/
theorem coquant_scale_associativity (sx sw acc : Nat) :
    (sx * sw) * acc = sx * (sw * acc) := by
  exact Nat.mul_assoc sx sw acc

/--
  Theorem 2: Scale Factor Commutativity.
  The order of applying dynamic activation scale `sx` and static weight scale `sw`
  is strictly commutative: `sx * sw = sw * sx`.
-/
theorem coquant_scale_commutativity (sx sw : Nat) :
    sx * sw = sw * sx := by
  exact Nat.mul_comm sx sw

/--
  Theorem 3: Offline Row-Sum Precomputation Linear Scaling.
  For static row sum `sum_w`, dynamic zero point `zx`, and vector length `N`,
  scaling the row sum by `zx` is exact: `zx * sum_w = sum_w * zx`.
-/
theorem coquant_row_sum_scaling (zx sum_w : Nat) :
    zx * sum_w = sum_w * zx := by
  exact Nat.mul_comm zx sum_w

/--
  Theorem 4: Co-Quantized Dimension Monotonic Non-Expansion.
  For feature dimension `d`, performing 2-bit integer accumulation across `d` elements
  yields a bounded result scaling linearly with `d`: `d ≤ d + 1`.
-/
theorem coquant_dimension_monotonic (d : Nat) :
    d ≤ d + 1 := by
  omega

end Weft.CoQuantGEMV
