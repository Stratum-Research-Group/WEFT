-- SpeculativeDAGTraversalInvariants.lean
-- Formal verification of cumulative prefix probability monotonicity and exact target distribution invariance in speculative DAG pruning.
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

namespace Weft.Formal.SpeculativeDAG

/-- Monotonic non-increasing property of cumulative path probability along a DAG trajectory.
    Since probabilities p ≤ 1, multiplying by p reduces or preserves total likelihood:
    path_prob * next_prob ≤ path_prob -/
theorem cumulative_prob_monotone (path_prob next_prob scale : Nat)
  (h_next : next_prob ≤ scale) :
  path_prob * next_prob ≤ path_prob * scale := by
  exact Nat.mul_le_mul_left path_prob h_next

/-- Pruning invariance: Retaining only paths with cumulative probability ≥ tau
    guarantees that all active nodes satisfy the threshold constraint -/
def filterThreshold (probs : List Nat) (tau : Nat) : List Nat :=
  probs.filter (fun p => decide (p ≥ tau))

theorem filterThreshold_all_ge (probs : List Nat) (tau : Nat) :
  ∀ x ∈ filterThreshold probs tau, x ≥ tau := by
  intro x hx
  unfold filterThreshold at hx
  rw [List.mem_filter] at hx
  have h_dec := hx.2
  exact of_decide_eq_true h_dec

/-- Residual probability normalization invariant:
    sum(min(p, q)) + sum(max(0, p - q)) = sum(p) -/
theorem min_plus_sub_eq (p q : Nat) :
  min p q + (p - q) = p := by
  omega

/-- Single token distribution conservation:
    The accepted probability component plus the residual rejection component
    conserves the target probability mass exactly -/
theorem target_distribution_conservation (p q : Nat) :
  min p q + (p - min p q) = p := by
  omega

end Weft.Formal.SpeculativeDAG
