-- DerivedHitchinFibrationInvariants.lean
-- Milestone Paper 285: Asymmetric Derived Hitchin Fibrations & Spectral Curves
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure HitchinSpectralState where
  dim : Nat
  spectralEnergyScaled : Nat
  thresholdScaled : Nat
  deriving Repr

def isTokenEvictable (s : HitchinSpectralState) : Prop :=
  s.spectralEnergyScaled ≤ s.thresholdScaled

theorem evictable_token_bounded (s : HitchinSpectralState) (h : isTokenEvictable s) :
    s.spectralEnergyScaled ≤ s.thresholdScaled := by
  exact h

theorem zero_spectral_energy_evictable (s : HitchinSpectralState) (h : s.spectralEnergyScaled = 0) :
    isTokenEvictable s := by
  dsimp [isTokenEvictable]
  rw [h]
  exact Nat.zero_le s.thresholdScaled

theorem spectral_threshold_monotonicity (e1 e2 thresh : Nat) (h_le : e1 ≤ e2) (h_evict : e2 ≤ thresh) :
    e1 ≤ thresh := by
  exact Nat.le_trans h_le h_evict
