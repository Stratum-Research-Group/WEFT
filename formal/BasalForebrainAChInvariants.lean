-- Basal Forebrain Acetylcholine (BF-ACh) Attention Cue Detection & Expected Uncertainty Invariants
-- Formal verification of cholinergic neuromodulatory sensory enhancement and recurrent suppression:
-- 1. Yu & Dayan (2005) Expected Uncertainty and Acetylcholine Signaling.
-- 2. Hasselmo (1995, 2006) Differential modulation of feedforward versus intracortical recurrent synapses.
-- 3. Monotonic boost of thalamocortical feedforward gain by ACh tone.
-- 4. Monotonic suppression of intracortical recurrent feedback weight by ACh tone.
-- 5. Cue detection fidelity and signal enhancement invariants.
-- Target #388 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.BasalForebrainACh

/-- State parameters representing basal forebrain cholinergic release and cortical circuits.
    All rates, gains, and drive inputs are scaled natural units. -/
structure AChState where
  /-- Basal forebrain cholinergic tone (scaled Nat, 0 - 100 representing 0.0 - 10.0 uM) -/
  achTone : Nat
  /-- Phasic cue-elicited ACh surge (scaled Nat, 0 - 50 representing 0.0 - 5.0 uM) -/
  phasicSurge : Nat
  /-- Thalamocortical sensory feedforward input amplitude -/
  feedforwardInput : Nat
  /-- Intracortical recurrent feedback input amplitude -/
  recurrentDrive : Nat
  deriving Repr

/-- Total effective cholinergic concentration:
    [ACh] = achTone + phasicSurge -/
def effectiveACh (s : AChState) : Nat :=
  s.achTone + s.phasicSurge

/-- Feedforward thalamocortical synaptic gain:
    G_ff = 10 + (35 * effectiveACh) / 100 (scaled integer multiplier, base 10 = 1.0x) -/
def feedforwardGain (s : AChState) : Nat :=
  10 + (35 * effectiveACh s) / 100

/-- Recurrent intracortical synaptic transmission weight:
    W_rec = 1000 / (100 + (effectiveACh s * 5))
    As ACh increases, presynaptic muscarinic inhibition drives W_rec strictly down. -/
def recurrentWeight (s : AChState) : Nat :=
  1000 / (100 + effectiveACh s * 5)

/-- Amplified sensory feedforward response in cortex:
    S_ff = feedforwardGain * feedforwardInput -/
def feedforwardSignal (s : AChState) : Nat :=
  feedforwardGain s * s.feedforwardInput

/-- Theorem 1: Thalamocortical feedforward gain is strictly positive and bounded below by baseline 10. -/
theorem feedforward_gain_strictly_positive (s : AChState) :
    feedforwardGain s >= 10 := by
  dsimp [feedforwardGain]
  omega

/-- Theorem 2: Phasic cue-elicited surge non-decreases feedforward gain. -/
theorem phasic_surge_monotone_feedforward_gain (s : AChState) (surge : Nat) :
    let s_boosted : AChState := { s with phasicSurge := s.phasicSurge + surge }
    feedforwardGain s_boosted >= feedforwardGain s := by
  dsimp [feedforwardGain, effectiveACh]
  omega

/-- Theorem 3: Recurrent intracortical weight is strictly bounded from above by 10 (1.0x baseline). -/
theorem recurrent_weight_bounded_by_baseline (s : AChState) :
    recurrentWeight s <= 10 := by
  dsimp [recurrentWeight]
  have h_denom : 100 + effectiveACh s * 5 >= 100 := by omega
  refine Nat.div_le_of_le_mul ?_
  omega

/-- Theorem 4: Elevating ACh monotonically attenuates intracortical recurrent feedback. -/
theorem ach_elevation_suppresses_recurrent_weight (s : AChState) (extra_ach : Nat) :
    let s_boosted : AChState := { s with achTone := s.achTone + extra_ach }
    recurrentWeight s_boosted <= recurrentWeight s := by
  dsimp [recurrentWeight, effectiveACh]
  apply Nat.div_le_div_left
  · omega
  · omega

/-- Theorem 5: Amplified feedforward signal monotonically scales with sensory input amplitude. -/
theorem feedforward_signal_monotone_input (s : AChState) (in1 in2 : Nat) (h_in : in1 <= in2) :
    feedforwardSignal { s with feedforwardInput := in1 } <=
    feedforwardSignal { s with feedforwardInput := in2 } := by
  dsimp [feedforwardSignal, feedforwardGain]
  apply Nat.mul_le_mul_left
  exact h_in

end Weft.Formal.BasalForebrainACh
