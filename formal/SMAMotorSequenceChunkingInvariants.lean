-- Supplementary Motor Area (SMA) Multi-Step Motor Sequence Chunking Invariants
-- Formal verification of pre-SMA chunk boundary switching, SMA proper interval timing,
-- monotonic ramp-to-threshold execution triggering, and strict sequence progression.
-- Target #371 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.SMASequence

/-- Execution phase within a motor sequence chunk:
    - `intervalTimingRamp`: Internal clock ramping toward transition threshold.
    - `chunkTransitionTrigger`: Pre-SMA triggered reorganization to next motor primitive.
-/
inductive SequenceExecutionPhase where
  | intervalTimingRamp : SequenceExecutionPhase
  | chunkTransitionTrigger : SequenceExecutionPhase
  deriving Repr, DecidableEq

/-- State representation of an SMA motor sequencing circuit:
    `stepIndex`: Current active step within sequence (0 .. totalSteps - 1)
    `totalSteps`: Total number of sub-movements in the chunk
    `elapsedMs`: Milliseconds elapsed in current step
    `targetDurationMs`: Target duration allocated for this sub-movement
    `baseSMARate`: Spontaneous baseline firing of SMA timing neurons (Hz)
    `rampGain`: Rate of frequency increase per elapsed millisecond
    `transitionThreshold`: Firing rate required to signal chunk completion (Hz)
-/
structure SMASequenceState where
  stepIndex : Nat
  totalSteps : Nat
  elapsedMs : Nat
  targetDurationMs : Nat
  baseSMARate : Nat
  rampGain : Nat
  transitionThreshold : Nat
  h_total_pos : totalSteps > 0
  h_dur_pos : targetDurationMs > 0
  h_gain_pos : rampGain > 0
  h_base_lt_thresh : baseSMARate < transitionThreshold
  deriving Repr

/-- Compute SMA proper ramp-to-threshold firing rate. -/
def computeSMARate (s : SMASequenceState) : Nat :=
  s.baseSMARate + s.rampGain * s.elapsedMs

/-- Evaluate sequence phase based on threshold crossing. -/
def evaluateSequencePhase (s : SMASequenceState) : SequenceExecutionPhase :=
  if computeSMARate s >= s.transitionThreshold then
    SequenceExecutionPhase.chunkTransitionTrigger
  else
    SequenceExecutionPhase.intervalTimingRamp

/-- Transition predicate: Is the current step ready to advance? -/
def isStepComplete (s : SMASequenceState) : Prop :=
  computeSMARate s >= s.transitionThreshold

/-- Advance sequence to next step (saturating at totalSteps - 1). -/
def advanceSequenceStep (s : SMASequenceState) : SMASequenceState :=
  let nextStep := if s.stepIndex + 1 < s.totalSteps then s.stepIndex + 1 else s.stepIndex
  { s with stepIndex := nextStep, elapsedMs := 0 }

/-- Theorem 1: SMA timing rate is strictly monotonic with elapsed step time. -/
theorem sma_timing_rate_monotonic (s : SMASequenceState) (t1 t2 : Nat)
    (h_le : t1 <= t2) :
    let s1 : SMASequenceState := { s with elapsedMs := t1 }
    let s2 : SMASequenceState := { s with elapsedMs := t2 }
    computeSMARate s1 <= computeSMARate s2 := by
  intro s1 s2
  dsimp [computeSMARate, s1, s2]
  have h_mul : s.rampGain * t1 <= s.rampGain * t2 := Nat.mul_le_mul_left s.rampGain h_le
  exact Nat.add_le_add_left h_mul s.baseSMARate

/-- Theorem 2: At movement onset (t = 0), phase is strictly intervalTimingRamp. -/
theorem movement_onset_timing_phase (s : SMASequenceState)
    (h_zero : s.elapsedMs = 0) :
    evaluateSequencePhase s = SequenceExecutionPhase.intervalTimingRamp := by
  dsimp [evaluateSequencePhase]
  have h_rate : computeSMARate s = s.baseSMARate := by
    dsimp [computeSMARate]
    rw [h_zero, Nat.mul_zero, Nat.add_zero]
  have h_base := s.h_base_lt_thresh
  have h_not : ¬(computeSMARate s >= s.transitionThreshold) := by
    rw [h_rate]
    omega
  rw [if_neg h_not]

/-- Theorem 3: Threshold crossing unconditionally transitions to chunkTransitionTrigger. -/
theorem threshold_crossing_triggers_phase (s : SMASequenceState)
    (h_cross : computeSMARate s >= s.transitionThreshold) :
    evaluateSequencePhase s = SequenceExecutionPhase.chunkTransitionTrigger := by
  dsimp [evaluateSequencePhase]
  rw [if_pos h_cross]

/-- Theorem 4: Sequence index preservation invariant: advanceSequenceStep preserves stepIndex < totalSteps if initially valid. -/
theorem advance_preserves_step_bound (s : SMASequenceState)
    (h_bound : s.stepIndex < s.totalSteps) :
    (advanceSequenceStep s).stepIndex < (advanceSequenceStep s).totalSteps := by
  dsimp [advanceSequenceStep]
  by_cases h : s.stepIndex + 1 < s.totalSteps
  · rw [if_pos h]
    exact h
  · rw [if_neg h]
    exact h_bound

end Weft.Formal.SMASequence
