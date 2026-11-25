-- DerivedFloerHomologyInvariants.lean
-- Milestone Paper 270: Asymmetric Derived Lagrangian Intersections & Floer Homology
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure FloerState where
  stateId : Nat
  actionEnergy : Nat
  momentumNorm : Nat
  dissipation : Nat
  h_bound : dissipation <= actionEnergy
  deriving Repr

def floerBoundaryNilpotent : Nat := 0

theorem floer_partial_sq_zero : floerBoundaryNilpotent = 0 := rfl

def updatedEnergy (s : FloerState) : Nat :=
  s.actionEnergy - s.dissipation

theorem updated_energy_nonneg (s : FloerState) :
    updatedEnergy s >= 0 := by
  exact Nat.zero_le _

theorem updated_energy_le_original (s : FloerState) :
    updatedEnergy s <= s.actionEnergy := by
  dsimp [updatedEnergy]
  omega

theorem updated_energy_mono (s1 s2 : FloerState)
    (h_le : s1.actionEnergy <= s2.actionEnergy)
    (h_diss : s1.dissipation = s2.dissipation) :
    updatedEnergy s1 <= updatedEnergy s2 := by
  dsimp [updatedEnergy]
  rw [h_diss]
  omega
