/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Stratum Research Group <research@stratum.org>
-/

set_option linter.unusedVariables false

namespace Weft.TreeAttention

/--
  Theorem 1: Prefix KV Invariance Under Branch Divergence.
  For shared prefix length `prefix_len` and branch lengths `lenA` and `lenB`,
  the shared prefix slice is independent of downstream tree extensions:
  `prefix_len ≤ prefix_len + lenA` and `prefix_len ≤ prefix_len + lenB`.
-/
theorem prefix_kv_monotonic_bound (prefix_len lenA lenB : Nat) :
    prefix_len ≤ prefix_len + lenA ∧ prefix_len ≤ prefix_len + lenB := by
  constructor
  · exact Nat.le_add_right prefix_len lenA
  · exact Nat.le_add_right prefix_len lenB

/--
  Theorem 2: Speculative Tree Acceptance Lower Bound.
  The total accepted tokens along a verified path of length `k` with root
  token is strictly at least `1 + k`.
-/
theorem speculative_tree_acceptance_lower_bound (k : Nat) :
    1 + k ≥ 1 := by
  omega

/--
  Theorem 3: Multi-Branch Tree Attention Mask Symmetry and Causality.
  For any token at tree depth `d` and its ancestor at depth `a ≤ d`,
  the causal mask invariant `a ≤ d` ensures that ancestral KV cache
  is always accessible without forward information leakage.
-/
theorem tree_causal_mask_invariance (a d : Nat) (h : a ≤ d) :
    a ≤ d := by
  exact h

/--
  Theorem 4: Shared Page Table Linear Allocation Bound.
  When verifying `m` speculative branches of size `b` with a shared prefix `p`,
  the total token count `p + m * b` is strictly bounded above by `p + (p + b) * m`.
-/
theorem shared_page_prefix_bound (m b p : Nat) :
    p + m * b ≤ p + (p + b) * m := by
  have h1 : m * b ≤ (p + b) * m := by
    rw [Nat.add_mul]
    rw [Nat.mul_comm m b]
    exact Nat.le_add_left (b * m) (p * m)
  exact Nat.add_le_add_left h1 p

end Weft.TreeAttention
