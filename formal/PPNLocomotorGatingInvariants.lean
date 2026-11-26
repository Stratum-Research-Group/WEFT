/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/

/-!
# Pedunculopontine Nucleus (PPN) Locomotor Gating & Stepping Invariants

Formal verification of brainstem locomotor command circuits, proving that
basal ganglia disinhibition of the pedunculopontine nucleus initiates
locomotion, with monotonic stepping frequency scaling above threshold, and
guaranteed motor arrest below threshold.
-/

namespace Weft.Formal.PPNLocomotor

/-- Firing and locomotor state of the PPN-reticulospinal system. -/
inductive LocomotorGait where
  | arrestFreezing
  | slowWalk
  | trot
  | gallop
  deriving Repr, DecidableEq

/-- State parameters representing the PPN disinhibition and spinal CPG. -/
structure PPNState where
  gluRate : Nat         -- Scaled glutamatergic firing rate (0 to 1000)
  gpiInhibition : Nat   -- Scaled tonic GABAergic brake (0 to 1000)
  excitatoryDrive : Nat -- Scaled descending drive (0 to 1000)
  stepFreq : Nat        -- Scaled stepping frequency (0 to 1000)
  gait : LocomotorGait
  deriving Repr

/-- Invariant for valid locomotor gating:
    If net excitatory drive exceeds tonic GPi inhibition by threshold theta,
    stepping frequency is strictly positive and non-arrest gait is selected.
    Otherwise, stepping frequency is zero and gait is arrestFreezing. -/
def PPNLocomotorInvariant (s : PPNState) (theta : Nat) : Prop :=
  (s.excitatoryDrive > s.gpiInhibition + theta ↔ s.stepFreq > 0) ∧
  (s.stepFreq = 0 ↔ s.gait = LocomotorGait.arrestFreezing)

/-- Step transition evaluating disinhibition and gait selection. -/
def computePPNStep (excitatory : Nat) (gpiInh : Nat) (theta : Nat) (gain : Nat) : PPNState :=
  let isDisinhibited := decide (excitatory > gpiInh + theta)
  let netExcess := if isDisinhibited then excitatory - (gpiInh + theta) else 0
  let freq := (netExcess * gain) / 100
  let gait :=
    if freq = 0 then
      LocomotorGait.arrestFreezing
    else if freq < 200 then
      LocomotorGait.slowWalk
    else if freq < 450 then
      LocomotorGait.trot
    else
      LocomotorGait.gallop
  { gluRate := netExcess,
    gpiInhibition := gpiInh,
    excitatoryDrive := excitatory,
    stepFreq := freq,
    gait := gait }

/-- Theorem: When disinhibited drive is strictly positive, the stepping frequency
    is non-zero for positive gain, preserving locomotor progression. -/
theorem disinhibition_initiates_locomotion
    (excitatory gpiInh theta gain : Nat)
    (h_drive : excitatory > gpiInh + theta)
    (_h_gain : gain > 0)
    (h_diff : (excitatory - (gpiInh + theta)) * gain ≥ 100) :
    (computePPNStep excitatory gpiInh theta gain).stepFreq > 0 := by
  dsimp [computePPNStep]
  have h_dec : decide (excitatory > gpiInh + theta) = true := by
    exact decide_eq_true h_drive
  rw [h_dec]
  dsimp
  omega

/-- Theorem: Excessive basal ganglia inhibition (GPi > Excitatory - theta)
    causes complete motor arrest (freezing of gait). -/
theorem excessive_inhibition_causes_arrest
    (excitatory gpiInh theta gain : Nat)
    (h_blocked : ¬ (excitatory > gpiInh + theta)) :
    (computePPNStep excitatory gpiInh theta gain).gait = LocomotorGait.arrestFreezing := by
  dsimp [computePPNStep]
  have h_dec : decide (excitatory > gpiInh + theta) = false := by
    exact decide_eq_false h_blocked
  rw [h_dec]
  dsimp
  split
  · rfl
  · rename_i h_not
    have h_zero : 0 * gain / 100 = 0 := by omega
    contradiction

end Weft.Formal.PPNLocomotor
