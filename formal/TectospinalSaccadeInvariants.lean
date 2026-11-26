-- Tectospinal Saccade Invariants and Superior Colliculus Retinotopic Population Vectors
-- Formal verification of the deep layers of the superior colliculus (DLSC) logarithmic motor map,
-- tectospinal tract cervical motor pool projection, omnipause neuron (OPN) gating,
-- and coordinated eye-head gaze shift decomposition.
-- Target #367 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.TectospinalSaccade

/-- Gaze orienting execution mode:
    - `saccadeEngaged`: Active burst orientation displacing visual line of sight.
    - `fixationMaintained`: Omnipause neurons gating motor burst.
-/
inductive CollicularGazeMode where
  | saccadeEngaged : CollicularGazeMode
  | fixationMaintained : CollicularGazeMode
  deriving Repr, DecidableEq

/-- State parameters for superior colliculus motor burst and eye-head gaze shift:
    `targetEccentricity`: Visual target eccentricity (degrees, scaled as Nat)
    `headThreshold`: Head movement recruitment threshold (degrees, e.g. 15 deg)
    `collicularGain`: Firing gain of collicular burst generation
    `tectoGain`: Cervical tectospinal coupling gain
    `opnInhibitionThreshold`: Omnipause neuron inhibition threshold
    `eyeGain`: Saccadic eye velocity gain
-/
structure CollicularGazeState where
  targetEccentricity : Nat
  headThreshold : Nat
  collicularGain : Nat
  tectoGain : Nat
  opnInhibitionThreshold : Nat
  eyeGain : Nat
  h_coll_pos : collicularGain > 0
  h_tecto_pos : tectoGain > 0
  h_eye_pos : eyeGain > 0
  deriving Repr

/-- Compute DLSC population burst frequency (Hz). -/
def computeCollicularBurst (s : CollicularGazeState) : Nat :=
  s.collicularGain * s.targetEccentricity

/-- Compute eye saccade command velocity. -/
def computeEyeVelocity (s : CollicularGazeState) : Nat :=
  s.eyeGain * s.targetEccentricity

/-- Compute cervical head turn velocity command.
    Head turn only engages if target eccentricity exceeds headThreshold. -/
def computeHeadVelocity (s : CollicularGazeState) : Nat :=
  if s.targetEccentricity >= s.headThreshold then
    s.tectoGain * (s.targetEccentricity - s.headThreshold)
  else
    0

/-- Evaluate collicular gaze mode based on OPN threshold crossing. -/
def evaluateGazeMode (s : CollicularGazeState) : CollicularGazeMode :=
  if computeCollicularBurst s >= s.opnInhibitionThreshold then
    CollicularGazeMode.saccadeEngaged
  else
    CollicularGazeMode.fixationMaintained

/-- Saccade initiation condition: collicular burst must overcome omnipause neuron inhibition. -/
def saccadeInitiationPermitted (s : CollicularGazeState) : Prop :=
  computeCollicularBurst s >= s.opnInhibitionThreshold

/-- Combined gaze shift velocity command: sum of eye and head contributions. -/
def computeTotalGazeVelocity (s : CollicularGazeState) : Nat :=
  computeEyeVelocity s + computeHeadVelocity s

/-- Theorem 1: Saccade burst monotonicity with respect to target eccentricity. -/
theorem saccade_burst_monotonic (s : CollicularGazeState) (e1 e2 : Nat)
    (h_le : e1 <= e2) :
    let s1 : CollicularGazeState := { s with targetEccentricity := e1 }
    let s2 : CollicularGazeState := { s with targetEccentricity := e2 }
    computeCollicularBurst s1 <= computeCollicularBurst s2 := by
  intro s1 s2
  dsimp [computeCollicularBurst, s1, s2]
  exact Nat.mul_le_mul_left s.collicularGain h_le

/-- Theorem 2: Head movement recruitment threshold invariant.
    For target eccentricity below head threshold, head velocity is strictly zero. -/
theorem head_velocity_below_threshold (s : CollicularGazeState)
    (h_lt : s.targetEccentricity < s.headThreshold) :
    computeHeadVelocity s = 0 := by
  dsimp [computeHeadVelocity]
  have h_not : ¬(s.targetEccentricity >= s.headThreshold) := by omega
  rw [if_neg h_not]

/-- Theorem 3: Head movement recruitment above threshold is monotonic. -/
theorem head_velocity_above_threshold_monotonic (s : CollicularGazeState) (e1 e2 : Nat)
    (h_th1 : e1 >= s.headThreshold)
    (h_le : e1 <= e2) :
    let s1 : CollicularGazeState := { s with targetEccentricity := e1 }
    let s2 : CollicularGazeState := { s with targetEccentricity := e2 }
    computeHeadVelocity s1 <= computeHeadVelocity s2 := by
  intro s1 s2
  dsimp [computeHeadVelocity, s1, s2]
  have h_th2 : e2 >= s.headThreshold := by omega
  have h_pos1 : e1 >= s.headThreshold := h_th1
  rw [if_pos h_pos1]
  rw [if_pos h_th2]
  have h_sub : e1 - s.headThreshold <= e2 - s.headThreshold := by omega
  exact Nat.mul_le_mul_left s.tectoGain h_sub

/-- Theorem 4: Head movement velocity monotonicity generally for any e1 <= e2. -/
theorem head_velocity_monotonic (s : CollicularGazeState) (e1 e2 : Nat)
    (h_le : e1 <= e2) :
    let s1 : CollicularGazeState := { s with targetEccentricity := e1 }
    let s2 : CollicularGazeState := { s with targetEccentricity := e2 }
    computeHeadVelocity s1 <= computeHeadVelocity s2 := by
  intro s1 s2
  by_cases h1 : e1 >= s.headThreshold
  · exact head_velocity_above_threshold_monotonic s e1 e2 h1 h_le
  · have h_lt1 : e1 < s.headThreshold := by omega
    have h_zero1 : computeHeadVelocity s1 = 0 := by
      dsimp [computeHeadVelocity, s1]
      have h_not : ¬(e1 >= s.headThreshold) := by omega
      rw [if_neg h_not]
    rw [h_zero1]
    exact Nat.zero_le _

/-- Theorem 5: Total gaze velocity monotonicity with respect to eccentricity. -/
theorem total_gaze_velocity_monotonic (s : CollicularGazeState) (e1 e2 : Nat)
    (h_le : e1 <= e2) :
    let s1 : CollicularGazeState := { s with targetEccentricity := e1 }
    let s2 : CollicularGazeState := { s with targetEccentricity := e2 }
    computeTotalGazeVelocity s1 <= computeTotalGazeVelocity s2 := by
  intro s1 s2
  dsimp [computeTotalGazeVelocity, s1, s2]
  have h_eye : computeEyeVelocity s1 <= computeEyeVelocity s2 := by
    dsimp [computeEyeVelocity, s1, s2]
    exact Nat.mul_le_mul_left s.eyeGain h_le
  have h_head : computeHeadVelocity s1 <= computeHeadVelocity s2 :=
    head_velocity_monotonic s e1 e2 h_le
  exact Nat.add_le_add h_eye h_head

/-- Theorem 6: Saccade initiation gate preservation.
    If eccentricity generates suprathreshold burst, mode is saccadeEngaged. -/
theorem saccade_initiation_mode_gate (s : CollicularGazeState)
    (h_burst : computeCollicularBurst s >= s.opnInhibitionThreshold) :
    evaluateGazeMode s = CollicularGazeMode.saccadeEngaged := by
  dsimp [evaluateGazeMode]
  rw [if_pos h_burst]

end Weft.Formal.TectospinalSaccade
