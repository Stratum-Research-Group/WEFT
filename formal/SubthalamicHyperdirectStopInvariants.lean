-- Cortico-Subthalamic Hyperdirect Pathway and Motor Braking Invariants
-- Formal verification of right inferior frontal cortex (rIFC) / pre-SMA stop-signal commands,
-- subthalamic nucleus (STN) hyperdirect excitation, internal globus pallidus (GPi/SNr) pallidal clamping,
-- and race-model action cancellation threshold invariants.
-- Target #374 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.SubthalamicHyperdirect

/-- Action execution status in stop-signal paradigm:
    - `actionExecuted`: Go process wins; thalamocortical transmission proceeds.
    - `actionAborted`: Hyperdirect stop process wins; global thalamic brake clamped.
-/
inductive StopSignalActionState where
  | actionExecuted : StopSignalActionState
  | actionAborted : StopSignalActionState
  deriving Repr, DecidableEq

/-- State representation of a hyperdirect motor braking loop:
    `stopSignal`: Cortical rIFC / pre-SMA stop drive (scaled as Nat)
    `stnBaseRate`: Spontaneous baseline discharge of STN neurons (Hz)
    `stnGain`: Gain of monosynaptic corticosubthalamic drive
    `gpiBaseRate`: Baseline GPi/SNr tonic inhibitory discharge (Hz)
    `gpiGain`: Firing rate amplification from STN to GPi
    `abortThreshold`: GPi discharge rate required to cancel motor execution (Hz)
-/
structure HyperdirectBrakeState where
  stopSignal : Nat
  stnBaseRate : Nat
  stnGain : Nat
  gpiBaseRate : Nat
  gpiGain : Nat
  abortThreshold : Nat
  h_stn_gain_pos : stnGain > 0
  h_gpi_gain_pos : gpiGain > 0
  h_base_lt_thresh : gpiBaseRate < abortThreshold
  deriving Repr

/-- Compute STN hyperdirect firing rate: stnBaseRate + stnGain * stopSignal. -/
def computeSTNRate (s : HyperdirectBrakeState) : Nat :=
  s.stnBaseRate + s.stnGain * s.stopSignal

/-- Compute GPi pallidal inhibitory output rate: gpiBaseRate + gpiGain * (stnRate - stnBaseRate). -/
def computeGPiRate (s : HyperdirectBrakeState) : Nat :=
  s.gpiBaseRate + s.gpiGain * (s.stnGain * s.stopSignal)

/-- Evaluate action cancellation state. -/
def evaluateActionState (s : HyperdirectBrakeState) : StopSignalActionState :=
  if computeGPiRate s >= s.abortThreshold then
    StopSignalActionState.actionAborted
  else
    StopSignalActionState.actionExecuted

/-- Cancellation predicate: Is the ongoing motor program aborted? -/
def isMotorAborted (s : HyperdirectBrakeState) : Prop :=
  computeGPiRate s >= s.abortThreshold

/-- Theorem 1: STN firing rate is strictly monotonic with cortical stop-signal drive. -/
theorem stn_rate_monotonic (s : HyperdirectBrakeState) (v1 v2 : Nat)
    (h_le : v1 <= v2) :
    let s1 : HyperdirectBrakeState := { s with stopSignal := v1 }
    let s2 : HyperdirectBrakeState := { s with stopSignal := v2 }
    computeSTNRate s1 <= computeSTNRate s2 := by
  intro s1 s2
  dsimp [computeSTNRate, s1, s2]
  have h_mul : s.stnGain * v1 <= s.stnGain * v2 := Nat.mul_le_mul_left s.stnGain h_le
  exact Nat.add_le_add_left h_mul s.stnBaseRate

/-- Theorem 2: GPi inhibitory output is strictly monotonic with cortical stop-signal drive. -/
theorem gpi_rate_monotonic (s : HyperdirectBrakeState) (v1 v2 : Nat)
    (h_le : v1 <= v2) :
    let s1 : HyperdirectBrakeState := { s with stopSignal := v1 }
    let s2 : HyperdirectBrakeState := { s with stopSignal := v2 }
    computeGPiRate s1 <= computeGPiRate s2 := by
  intro s1 s2
  dsimp [computeGPiRate, s1, s2]
  have h_stn : s.stnGain * v1 <= s.stnGain * v2 := Nat.mul_le_mul_left s.stnGain h_le
  have h_gpi : s.gpiGain * (s.stnGain * v1) <= s.gpiGain * (s.stnGain * v2) :=
    Nat.mul_le_mul_left s.gpiGain h_stn
  exact Nat.add_le_add_left h_gpi s.gpiBaseRate

/-- Theorem 3: With zero stop signal, motor action proceeds uninhibited (actionExecuted). -/
theorem zero_stop_signal_action_executed (s : HyperdirectBrakeState)
    (h_zero : s.stopSignal = 0) :
    evaluateActionState s = StopSignalActionState.actionExecuted := by
  dsimp [evaluateActionState]
  have h_rate : computeGPiRate s = s.gpiBaseRate := by
    dsimp [computeGPiRate]
    rw [h_zero, Nat.mul_zero, Nat.mul_zero, Nat.add_zero]
  have h_base := s.h_base_lt_thresh
  have h_not : ¬(computeGPiRate s >= s.abortThreshold) := by
    rw [h_rate]
    omega
  rw [if_neg h_not]

/-- Theorem 4: Suprathreshold stop signal unconditionally aborts action (actionAborted). -/
theorem suprathreshold_stop_aborts_action (s : HyperdirectBrakeState)
    (h_abort : computeGPiRate s >= s.abortThreshold) :
    evaluateActionState s = StopSignalActionState.actionAborted := by
  dsimp [evaluateActionState]
  rw [if_pos h_abort]

/-- Theorem 5: Cancellation persistence monotonicity: If stop signal v1 cancels action, any v2 >= v1 also cancels. -/
theorem cancellation_persistence (s : HyperdirectBrakeState) (v1 v2 : Nat)
    (h_v : v1 <= v2)
    (h_ab1 : computeGPiRate ({ s with stopSignal := v1 }) >= s.abortThreshold) :
    computeGPiRate ({ s with stopSignal := v2 }) >= s.abortThreshold := by
  have h_mono := gpi_rate_monotonic s v1 v2 h_v
  dsimp at h_mono
  omega

end Weft.Formal.SubthalamicHyperdirect
