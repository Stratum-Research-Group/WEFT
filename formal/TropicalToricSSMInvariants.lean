-- TropicalToricSSMInvariants.lean
-- Milestone Paper 262: Asymmetric Tropical Toric Ideals & Graded Gröbner Bases
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure SSMState where
  dimId : Nat
  stateVal : Nat
  inputVal : Nat
  deriving Repr

def tropicalRecurrenceStep (h x a b : Nat) : Nat :=
  let branchA := h + a
  let branchB := x + b
  if branchA >= branchB then branchA else branchB

theorem tropical_step_nonneg (h x a b : Nat) :
    tropicalRecurrenceStep h x a b >= 0 := by
  exact Nat.zero_le _

theorem tropical_step_ge_branch_a (h x a b : Nat) :
    tropicalRecurrenceStep h x a b >= h + a := by
  dsimp [tropicalRecurrenceStep]
  split <;> omega

theorem tropical_step_ge_branch_b (h x a b : Nat) :
    tropicalRecurrenceStep h x a b >= x + b := by
  dsimp [tropicalRecurrenceStep]
  split <;> omega

theorem tropical_step_monotonic (h1 h2 x a b : Nat)
    (h_le : h1 <= h2) :
    tropicalRecurrenceStep h1 x a b <= tropicalRecurrenceStep h2 x a b := by
  dsimp [tropicalRecurrenceStep]
  split
  · split <;> omega
  · split <;> omega
