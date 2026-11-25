-- MotivicStackPruningInvariants.lean
-- Milestone Paper 272: Asymmetric Motivic Cohomological Operations on Deligne-Mumford Stacks
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure StackToken where
  tokenId : Nat
  inertiaVal : Nat
  deriving Repr

def isTokenRetained (t : StackToken) (threshold : Nat) : Prop :=
  t.inertiaVal >= threshold

theorem token_inertia_nonneg (t : StackToken) :
    t.inertiaVal >= 0 := by
  exact Nat.zero_le _

theorem retention_threshold_mono (t : StackToken) (th1 th2 : Nat)
    (h_le : th1 <= th2)
    (h_ret : isTokenRetained t th2) :
    isTokenRetained t th1 := by
  dsimp [isTokenRetained] at *
  omega

def pruneResidual (t : StackToken) (threshold : Nat) : Nat :=
  if t.inertiaVal >= threshold then 0 else t.inertiaVal

theorem prune_residual_bound (t : StackToken) (threshold : Nat) :
    pruneResidual t threshold <= threshold := by
  dsimp [pruneResidual]
  split <;> omega
