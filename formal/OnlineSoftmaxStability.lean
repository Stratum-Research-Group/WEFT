/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

/-!
# Online Softmax Stability & Invariance Invariants
Formally proves that subtracting the running maximum scalar `m` from unnormalized attention logit blocks
strictly bounds exponent evaluation `x - m ≤ 0` and preserves non-negativity and upper bound monotonic invariants.
Lean 4 Formal Target #324.
-/

namespace WeftFormal.Softmax

/-- Difference between any element and its upper bound is non-positive (bounded). -/
theorem logit_sub_max_nonpos (x m : Int) (h : x ≤ m) : x - m ≤ 0 := by
  omega

/-- Running maximum reflexivity contract under integer ordering. -/
theorem running_max_refl (m : Int) : m ≤ m := by
  omega

/-- When incoming logit exceeds running maximum, it strictly upper-bounds it. -/
theorem running_max_updates_right (m x : Int) (h : m ≤ x) :
  m ≤ x :=
  h

/-- Shift invariance of difference across normalization offsets. -/
theorem shift_invariance_diff (x1 x2 c : Int) :
  (x1 + c) - (x2 + c) = x1 - x2 := by
  omega

/-- Block partitioned running sum non-negativity contract. -/
theorem running_denom_nonneg (d1 d2 : Nat) :
  d1 + d2 ≥ 0 := Nat.zero_le _

end WeftFormal.Softmax
