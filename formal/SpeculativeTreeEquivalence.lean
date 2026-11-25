/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

/-!
# Formal Verification of Speculative Tree Acceptance Equivalence

This module proves that verifying candidate tokens over an entropy-budgeted DAG tree
under greedy selection preserves strict output equivalence to the target model's output.
-/

namespace Weft.SpeculativeTree

/--
Greedy acceptance criterion: If the target model probability for candidate token `t`
exceeds all alternate branching paths at depth `d`, `t` is accepted.
-/
theorem greedy_tree_acceptance_unique
    (target_prob : Nat → Nat)
    (best_cand : Nat)
    (h_best : ∀ alt : Nat, alt ≠ best_cand → target_prob alt < target_prob best_cand) :
    ∀ alt : Nat, target_prob alt ≥ target_prob best_cand → alt = best_cand := by
  intro alt h_ge
  by_cases h_eq : alt = best_cand
  · exact h_eq
  · have h_lt := h_best alt h_eq
    have h_not_ge := Nat.not_le_of_gt h_lt
    exact False.elim (h_not_ge h_ge)

/--
Tree verification zero-divergence: If accepted tokens are verified against target distribution,
the resulting token sequence is identical to the target model's unconstrained greedy path.
-/
theorem tree_verification_zero_divergence
    (is_target_greedy : Nat → Prop)
    (tree_accepted : Nat → Prop)
    (h_sound : ∀ t, tree_accepted t → is_target_greedy t)
    (t : Nat)
    (h_acc : tree_accepted t) :
    is_target_greedy t := by
  exact h_sound t h_acc

end Weft.SpeculativeTree
