-- Cerebellar Long-Term Depression (LTD) & Coincidence Convergence Invariants
-- Formal proof of heterosynaptic coincidence detection, error-driven parallel fiber weight reduction,
-- and asymptotic bounded plasticity.
-- Target #357 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.CerebellarLTD

/-- Synaptic plasticity state of a Parallel Fiber - Purkinje Cell synapse. -/
inductive SynapseState where
  | quiescent : SynapseState
  | ltp : SynapseState
  | ltd : SynapseState
  deriving Repr, DecidableEq

/-- State representation of a parallel fiber synapse:
    `weight`: Scaled weight in [100, 1000] (0.1 to 1.0)
    `pfActivity`: 1 if active, 0 otherwise
    `cfError`: 1 if climbing fiber fired error spike, 0 otherwise
    `wMin`: lower bound (e.g. 100)
    `wMax`: upper bound (e.g. 1000)
-/
structure Synapse where
  weight : Nat
  pfActivity : Nat
  cfError : Nat
  wMin : Nat
  wMax : Nat
  h_bounds : wMin ≤ weight ∧ weight ≤ wMax
  h_min_pos : wMin > 0
  h_order : wMin < wMax
  deriving Repr

/-- Coincidence condition: Both Parallel Fiber and Climbing Fiber are simultaneously active. -/
def isCoincident (s : Synapse) : Prop :=
  s.pfActivity > 0 ∧ s.cfError > 0

/-- Evaluate step-wise plasticity update:
    If coincident (PF active AND CF error), apply depression `eta_ltd`.
    If PF active AND NOT CF error, apply slight potentiation `eta_ltp`.
    Otherwise no change.
-/
def updateWeight (s : Synapse) (eta_ltd : Nat) (eta_ltp : Nat) : Nat :=
  if s.pfActivity > 0 ∧ s.cfError > 0 then
    if s.weight ≥ s.wMin + eta_ltd then s.weight - eta_ltd else s.wMin
  else if s.pfActivity > 0 ∧ s.cfError = 0 then
    if s.weight + eta_ltp ≤ s.wMax then s.weight + eta_ltp else s.wMax
  else
    s.weight

/-- Theorem: Under heterosynaptic coincidence (PF > 0 and CF > 0), the updated weight
    is strictly less than or equal to the original weight, and remains bounded by wMin. -/
theorem ltd_depresses_weight
    (s : Synapse) (eta_ltd eta_ltp : Nat)
    (h_coinc : isCoincident s) :
    let w_next := updateWeight s eta_ltd eta_ltp
    w_next ≤ s.weight ∧ w_next ≥ s.wMin := by
  dsimp [isCoincident] at h_coinc
  dsimp [updateWeight]
  have h_cond : s.pfActivity > 0 ∧ s.cfError > 0 := h_coinc
  rw [if_pos h_cond]
  split
  · rename_i h_ge
    constructor
    · omega
    · omega
  · rename_i h_not_ge
    constructor
    · rcases s.h_bounds with ⟨h_min, _⟩
      exact h_min
    · omega

/-- Theorem: Weight updates always preserve the invariant lower and upper bounds [wMin, wMax]. -/
theorem update_preserves_bounds
    (s : Synapse) (eta_ltd eta_ltp : Nat) :
    let w_next := updateWeight s eta_ltd eta_ltp
    s.wMin ≤ w_next ∧ w_next ≤ s.wMax := by
  dsimp [updateWeight]
  split
  · rename_i h_ltd
    split
    · rename_i h_ge
      rcases s.h_bounds with ⟨h_min, h_max⟩
      constructor
      · omega
      · omega
    · rename_i h_not_ge
      rcases s.h_bounds with ⟨h_min, h_max⟩
      constructor
      · omega
      · omega
  · rename_i h_not_ltd
    split
    · rename_i h_ltp
      split
      · rename_i h_le
        rcases s.h_bounds with ⟨h_min, h_max⟩
        constructor
        · omega
        · omega
      · rename_i h_not_le
        rcases s.h_bounds with ⟨h_min, h_max⟩
        constructor
        · omega
        · omega
    · rename_i h_no_change
      exact s.h_bounds

end Weft.Formal.CerebellarLTD
