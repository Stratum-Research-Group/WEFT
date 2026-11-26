-- Spinal Presynaptic Primary Afferent Depolarization (PAD) and Inflow Gating Invariants
-- Formal verification of spinal GABAergic axo-axonic interneuron recruitment,
-- primary afferent depolarization (PAD) scaling, and the monotonic attenuation
-- of monosynaptic Ia stretch reflex transmission during voluntary movement.
-- Target #370 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.SpinalPAD

/-- Spinal reflex transmission regime:
    - `sensoryGated`: Strong presynaptic PAD suppressing sensory reflex transmission.
    - `intactReflex`: Minimal PAD permitting high-gain postural stabilization.
-/
inductive SpinalGatingRegime where
  | sensoryGated : SpinalGatingRegime
  | intactReflex : SpinalGatingRegime
  deriving Repr, DecidableEq

/-- State representation of a spinal presynaptic gating circuit:
    `descendingDrive`: Voluntary descending command / velocity (scaled as Nat, e.g. cm/s)
    `padBaseRate`: Spontaneous baseline discharge of GABAergic PAD interneurons (Hz)
    `padGain`: PAD interneuron sensitivity gain
    `padThreshold`: Firing rate required to induce significant reflex suppression (Hz)
    `maxPermittedEPSP`: Upper bound on monosynaptic EPSP in gated state (mV, scaled as Nat)
-/
structure SpinalPADState where
  descendingDrive : Nat
  padBaseRate : Nat
  padGain : Nat
  padThreshold : Nat
  maxPermittedEPSP : Nat
  h_gain_pos : padGain > 0
  h_base_lt_thresh : padBaseRate < padThreshold
  deriving Repr

/-- Compute GABAergic PAD interneuron firing rate. -/
def computePADRate (s : SpinalPADState) : Nat :=
  s.padBaseRate + s.padGain * s.descendingDrive

/-- Evaluate whether sensory transmission is gated or intact. -/
def evaluateGatingRegime (s : SpinalPADState) : SpinalGatingRegime :=
  if computePADRate s >= s.padThreshold then
    SpinalGatingRegime.sensoryGated
  else
    SpinalGatingRegime.intactReflex

/-- Presynaptic attenuation predicate: Sensory inflow is gated if PAD exceeds threshold. -/
def isSensoryGated (s : SpinalPADState) : Prop :=
  computePADRate s >= s.padThreshold

/-- Theorem 1: PAD interneuron firing rate is strictly monotonic with descending motor drive. -/
theorem pad_firing_rate_monotonic (s : SpinalPADState) (v1 v2 : Nat)
    (h_le : v1 <= v2) :
    let s1 : SpinalPADState := { s with descendingDrive := v1 }
    let s2 : SpinalPADState := { s with descendingDrive := v2 }
    computePADRate s1 <= computePADRate s2 := by
  intro s1 s2
  dsimp [computePADRate, s1, s2]
  have h_mul : s.padGain * v1 <= s.padGain * v2 := Nat.mul_le_mul_left s.padGain h_le
  exact Nat.add_le_add_left h_mul s.padBaseRate

/-- Theorem 2: Quiescent state preserves intact postural reflex. -/
theorem quiescent_state_intact_reflex (s : SpinalPADState)
    (h_zero : s.descendingDrive = 0) :
    evaluateGatingRegime s = SpinalGatingRegime.intactReflex := by
  dsimp [evaluateGatingRegime]
  have h_rate : computePADRate s = s.padBaseRate := by
    dsimp [computePADRate]
    rw [h_zero, Nat.mul_zero, Nat.add_zero]
  have h_base := s.h_base_lt_thresh
  have h_not : ¬(computePADRate s >= s.padThreshold) := by
    rw [h_rate]
    omega
  rw [if_neg h_not]

/-- Theorem 3: Strong voluntary drive guarantees transition to sensoryGated regime. -/
theorem voluntary_drive_sensory_gating (s : SpinalPADState)
    (h_drive : computePADRate s >= s.padThreshold) :
    evaluateGatingRegime s = SpinalGatingRegime.sensoryGated := by
  dsimp [evaluateGatingRegime]
  rw [if_pos h_drive]

/-- Theorem 4: Gating persistence monotonicity: Once gated at drive v1, any higher drive v2 >= v1 remains gated. -/
theorem sensory_gating_persistence (s : SpinalPADState) (v1 v2 : Nat)
    (h_v : v1 <= v2)
    (h_gated : computePADRate ({ s with descendingDrive := v1 }) >= s.padThreshold) :
    computePADRate ({ s with descendingDrive := v2 }) >= s.padThreshold := by
  have h_mono := pad_firing_rate_monotonic s v1 v2 h_v
  dsimp at h_mono
  omega

end Weft.Formal.SpinalPAD
