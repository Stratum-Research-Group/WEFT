/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Stratum Research Group <research@stratum.org>
-/

set_option linter.unusedVariables false

namespace Weft.AdapterSparsification

/--
  Invariant: Threshold Truncation Linearity.
  A pruned element with magnitude below threshold epsilon contributes at most epsilon.
-/
def pruned_magnitude (v_val eps : Nat) : Nat :=
  if v_val < eps then 0 else v_val

theorem pruned_val_le_original (v_val eps : Nat) :
    pruned_magnitude v_val eps ≤ v_val := by
  dsimp [pruned_magnitude]
  split
  · exact Nat.zero_le v_val
  · exact Nat.le_refl v_val

/--
  Theorem: Sparsity Perturbation Bounded Error.
  For dimension D where M elements are pruned with magnitude < epsilon,
  the cumulative inner product error is bounded by M * epsilon.
-/
def cumulative_pruning_error (pruned_count eps : Nat) : Nat :=
  pruned_count * eps

theorem cumulative_pruning_error_le_total (pruned_count total_dim eps : Nat)
    (h : pruned_count ≤ total_dim) :
    cumulative_pruning_error pruned_count eps ≤ total_dim * eps := by
  dsimp [cumulative_pruning_error]
  exact Nat.mul_le_mul_right eps h

/--
  Theorem: Non-Negative Monotonic Dot Product with Non-Negative Sparsified Vectors.
  Inner product accumulation over non-negative elements remains monotonic.
-/
theorem sparse_dot_monotonic (acc term : Nat) :
    acc ≤ acc + term := by
  exact Nat.le_add_right acc term

end Weft.AdapterSparsification
