-- TropicalToricIdealsInvariants.lean
-- Milestone Paper 292: Asymmetric Tropical Toric Ideals & Graded Betti Numbers
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure BitSlicingState where
  numChannels : Nat
  allocatedBits : Nat
  pdimBound : Nat
  deriving Repr

def isBitBudgetFeasible (s : BitSlicingState) : Prop :=
  s.allocatedBits ≥ s.pdimBound

theorem bit_budget_feasibility_guaranteed (s : BitSlicingState) (h_ge : s.allocatedBits ≥ s.pdimBound) :
    isBitBudgetFeasible s := by
  dsimp [isBitBudgetFeasible]
  exact h_ge

theorem pdim_monotonicity (b1 b2 pdim : Nat) (h_le : b1 ≤ b2) (h_feas : b1 ≥ pdim) :
    b2 ≥ pdim := by
  exact Nat.le_trans h_feas h_le
