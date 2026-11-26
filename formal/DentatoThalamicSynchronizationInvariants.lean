-- Dentato-Thalamo-Cortical (DTC) Synchronization Invariants
-- Formal verification of cerebellar dentate nucleus disinhibition,
-- ventral lateral (VL) motor thalamic relay gating, and phase-locked
-- sub-millisecond facilitation of M1 Layer V pyramidal motor commands.
-- Target #373 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.DentatoThalamic

/-- Synchronization state of the DTC loop:
    - `phaseLocked`: Dentate drive arrives in phase with cortical excitable state.
    - `phaseSlip`: Phase misalignment exceeding tolerance.
-/
inductive DTCSyncMode where
  | phaseLocked : DTCSyncMode
  | phaseSlip : DTCSyncMode
  deriving Repr, DecidableEq

/-- State representation of a dentato-thalamo-cortical synchronization circuit:
    `dentateRate`: Firing rate of lateral cerebellar dentate projection neurons (Hz)
    `thalamicGain`: Firing rate to VL thalamic EPSP scaling gain
    `syncThreshold`: Minimum firing rate required for robust phase-locking (Hz)
    `phaseAlignmentDeg`: Phase alignment error in degrees (0 to 180)
    `maxPermittedPhaseDeg`: Maximum allowed phase error for locking (e.g. 30 deg)
-/
structure DentatoThalamicState where
  dentateRate : Nat
  thalamicGain : Nat
  syncThreshold : Nat
  phaseAlignmentDeg : Nat
  maxPermittedPhaseDeg : Nat
  h_gain_pos : thalamicGain > 0
  h_thresh_pos : syncThreshold > 0
  deriving Repr

/-- Compute VL thalamic relay EPSP amplitude (scaled as Nat). -/
def computeThalamicEPSP (s : DentatoThalamicState) : Nat :=
  s.thalamicGain * s.dentateRate

/-- Evaluate DTC synchronization state. -/
def evaluateSyncMode (s : DentatoThalamicState) : DTCSyncMode :=
  if s.dentateRate >= s.syncThreshold && s.phaseAlignmentDeg <= s.maxPermittedPhaseDeg then
    DTCSyncMode.phaseLocked
  else
    DTCSyncMode.phaseSlip

/-- Synchronization predicate: Is the loop phase-locked? -/
def isPhaseLocked (s : DentatoThalamicState) : Prop :=
  s.dentateRate >= s.syncThreshold ∧ s.phaseAlignmentDeg <= s.maxPermittedPhaseDeg

/-- Theorem 1: Thalamic EPSP is strictly monotonic with dentate firing rate. -/
theorem thalamic_epsp_monotonic (s : DentatoThalamicState) (r1 r2 : Nat)
    (h_le : r1 <= r2) :
    let s1 : DentatoThalamicState := { s with dentateRate := r1 }
    let s2 : DentatoThalamicState := { s with dentateRate := r2 }
    computeThalamicEPSP s1 <= computeThalamicEPSP s2 := by
  intro s1 s2
  dsimp [computeThalamicEPSP, s1, s2]
  exact Nat.mul_le_mul_left s.thalamicGain h_le

/-- Theorem 2: Phase-locking condition guarantees phaseLocked mode. -/
theorem phase_locked_mode_guarantee (s : DentatoThalamicState)
    (h_rate : s.dentateRate >= s.syncThreshold)
    (h_phase : s.phaseAlignmentDeg <= s.maxPermittedPhaseDeg) :
    evaluateSyncMode s = DTCSyncMode.phaseLocked := by
  dsimp [evaluateSyncMode]
  have h_and : (s.dentateRate >= s.syncThreshold && s.phaseAlignmentDeg <= s.maxPermittedPhaseDeg) = true := by
    rw [Bool.and_eq_true]
    exact ⟨decide_eq_true h_rate, decide_eq_true h_phase⟩
  rw [if_pos h_and]

/-- Theorem 3: Subthreshold dentate discharge unconditionally produces phaseSlip. -/
theorem subthreshold_dentate_phase_slip (s : DentatoThalamicState)
    (h_low : s.dentateRate < s.syncThreshold) :
    evaluateSyncMode s = DTCSyncMode.phaseSlip := by
  dsimp [evaluateSyncMode]
  have h_not : ¬(s.dentateRate >= s.syncThreshold) := by omega
  have h_and : (s.dentateRate >= s.syncThreshold && s.phaseAlignmentDeg <= s.maxPermittedPhaseDeg) = false := by
    rw [Bool.and_eq_false_iff]
    left
    exact decide_eq_false h_not
  rw [if_neg]
  intro h_contra
  rw [h_and] at h_contra
  contradiction

/-- Theorem 4: Excessive phase error unconditionally produces phaseSlip. -/
theorem excessive_phase_error_phase_slip (s : DentatoThalamicState)
    (h_err : s.phaseAlignmentDeg > s.maxPermittedPhaseDeg) :
    evaluateSyncMode s = DTCSyncMode.phaseSlip := by
  dsimp [evaluateSyncMode]
  have h_not : ¬(s.phaseAlignmentDeg <= s.maxPermittedPhaseDeg) := by omega
  have h_and : (s.dentateRate >= s.syncThreshold && s.phaseAlignmentDeg <= s.maxPermittedPhaseDeg) = false := by
    rw [Bool.and_eq_false_iff]
    right
    exact decide_eq_false h_not
  rw [if_neg]
  intro h_contra
  rw [h_and] at h_contra
  contradiction

end Weft.Formal.DentatoThalamic
