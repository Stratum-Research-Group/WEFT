-- TropicalToricFibersInvariants.lean
-- Milestone Paper 277: Asymmetric Tropical Toric Fibers & Amoeba Volumes
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure AmoebaFiberState where
  dim : Nat
  accumulatedDiff : Nat
  thresholdScaled : Nat
  deriving Repr

def isLayerSkippable (s : AmoebaFiberState) : Prop :=
  s.accumulatedDiff ≤ s.thresholdScaled

theorem skippable_layer_bounded (s : AmoebaFiberState) (h : isLayerSkippable s) :
    s.accumulatedDiff ≤ s.thresholdScaled := by
  exact h

theorem zero_divergence_skippable (s : AmoebaFiberState) (h : s.accumulatedDiff = 0) :
    isLayerSkippable s := by
  dsimp [isLayerSkippable]
  rw [h]
  exact Nat.zero_le s.thresholdScaled

theorem skip_decision_monotonicity (d1 d2 thresh : Nat) (h_le : d1 ≤ d2) (h_skip : d2 ≤ thresh) :
    d1 ≤ thresh := by
  exact Nat.le_trans h_le h_skip
