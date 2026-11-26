-- Neuromuscular Junction (NMJ) Quantal Safety Factor & EPP Invariants
-- Formal proof that quantal neurotransmitter release satisfying the biological safety factor
-- unconditionally elicits an all-or-none muscle action potential (100% transmission fidelity).
-- Target #360 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.NMJQuantal

/-- Muscle fiber transmission outcome:
    - `actionPotentialElicited`: Full sarcolemmal all-or-none depolarization and twitch.
    - `transmissionFailure`: Subthreshold EPP failing to evoke mechanical contraction.
-/
inductive NMJTransmissionOutcome where
  | actionPotentialElicited : NMJTransmissionOutcome
  | transmissionFailure : NMJTransmissionOutcome
  deriving Repr, DecidableEq

/-- State representation of an NMJ endplate:
    `quanta`: Number of ACh vesicles exocytosed (scaled Nat)
    `quantumSize`: Depolarization per vesicle in millivolts (e.g. 7 for 0.7 mV scaled by 10)
    `threshold`: Depolarization required to open Nav1.4 channels (e.g. 250 for 25.0 mV)
    `minSafetyFactorQuanta`: Minimum quanta guaranteeing safety factor >= 1.0
-/
structure NMJEndplateState where
  quanta : Nat
  quantumSize : Nat
  threshold : Nat
  minSafetyFactorQuanta : Nat
  h_thresh_pos : threshold > 0
  h_size_pos : quantumSize > 0
  h_safety_def : minSafetyFactorQuanta * quantumSize ≥ threshold
  deriving Repr

/-- Invariant for Healthy Mammalian NMJ Safety Factor:
    An endplate satisfies the safety factor invariant if the released quantal content
    meets or exceeds the minimum safety threshold (m >= minSafetyFactorQuanta).
-/
def satisfiesSafetyFactor (s : NMJEndplateState) : Prop :=
  s.quanta ≥ s.minSafetyFactorQuanta

/-- Evaluate endplate transmission outcome based on total Endplate Potential (EPP). -/
def evaluateNMJTransmission (s : NMJEndplateState) : NMJTransmissionOutcome :=
  let epp := s.quanta * s.quantumSize
  if epp ≥ s.threshold then
    NMJTransmissionOutcome.actionPotentialElicited
  else
    NMJTransmissionOutcome.transmissionFailure

/-- Theorem: For any NMJ endplate satisfying the biological safety factor invariant,
    the resulting Endplate Potential unconditionally elicits a muscle action potential. -/
theorem safety_factor_guarantees_muscle_ap
    (s : NMJEndplateState)
    (h_sf : satisfiesSafetyFactor s) :
    evaluateNMJTransmission s = NMJTransmissionOutcome.actionPotentialElicited := by
  dsimp [evaluateNMJTransmission]
  dsimp [satisfiesSafetyFactor] at h_sf
  have h_min := s.h_safety_def
  have h_mult : s.minSafetyFactorQuanta * s.quantumSize ≤ s.quanta * s.quantumSize :=
    Nat.mul_le_mul_right s.quantumSize h_sf
  have h_epp : s.quanta * s.quantumSize ≥ s.threshold := by omega
  rw [if_pos h_epp]

/-- Theorem: Below threshold quantal release strictly results in transmission failure. -/
theorem subthreshold_quanta_fails
    (s : NMJEndplateState)
    (h_sub : s.quanta * s.quantumSize < s.threshold) :
    evaluateNMJTransmission s = NMJTransmissionOutcome.transmissionFailure := by
  dsimp [evaluateNMJTransmission]
  have h_not_ge : ¬(s.quanta * s.quantumSize ≥ s.threshold) := by omega
  rw [if_neg h_not_ge]

end Weft.Formal.NMJQuantal
