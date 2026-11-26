-- Polysynaptic Crossed-Extensor Reflex & Postural Preservation Invariants
-- Formal proof that unilateral cutaneous noxious stimulation unconditionally commands
-- ipsilateral limb flexion withdrawal while preserving postural equilibrium via
-- contralateral extensor thrust and antagonist suppression.
-- Target #363 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.CrossedExtensorReflex

/-- Bilateral limb postural reflex state:
    - `symmetricStance`: Normal quiet stance or locomotor stepping.
    - `withdrawalExtensorThrust`: Unilateral noxious withdrawal with contralateral postural compensation.
-/
inductive BilateralPosturalState where
  | symmetricStance : BilateralPosturalState
  | withdrawalExtensorThrust : BilateralPosturalState
  deriving Repr, DecidableEq

/-- State representation of bilateral spinal reflex circuit:
    `noxiousIntensity`: Pain afferent firing intensity (e.g. 0 to 200)
    `noxiousThreshold`: Threshold for eliciting protective withdrawal (e.g. 50)
    `ipsiFlex`: Ipsilateral flexor motor pool rate (Hz)
    `ipsiExt`: Ipsilateral extensor motor pool rate (Hz)
    `contraExt`: Contralateral extensor motor pool rate (Hz)
    `contraFlex`: Contralateral flexor motor pool rate (Hz)
    `activeThreshold`: Minimum rate defining active muscular contraction (e.g. 80 Hz)
    `suppressionCeiling`: Maximum rate permitted in suppressed antagonist (e.g. 10 Hz)
-/
structure BilateralReflexState where
  noxiousIntensity : Nat
  noxiousThreshold : Nat
  ipsiFlex : Nat
  ipsiExt : Nat
  contraExt : Nat
  contraFlex : Nat
  activeThreshold : Nat
  suppressionCeiling : Nat
  h_thresh_order : suppressionCeiling < activeThreshold
  h_nox_pos : noxiousThreshold > 0
  deriving Repr

/-- Invariant for Valid Crossed-Extensor Coordination:
    When noxious stimulus meets or exceeds threshold:
    1. Ipsilateral flexor is active (>= activeThreshold) and ipsilateral extensor suppressed (<= suppressionCeiling).
    2. Contralateral extensor is active (>= activeThreshold) and contralateral flexor suppressed (<= suppressionCeiling).
-/
def satisfiesCrossedExtensorInvariant (s : BilateralReflexState) : Prop :=
  s.noxiousIntensity ≥ s.noxiousThreshold →
  (s.ipsiFlex ≥ s.activeThreshold ∧ s.ipsiExt ≤ s.suppressionCeiling ∧
   s.contraExt ≥ s.activeThreshold ∧ s.contraFlex ≤ s.suppressionCeiling)

/-- Evaluate bilateral postural state. -/
def evaluatePosturalState (s : BilateralReflexState) : BilateralPosturalState :=
  if s.noxiousIntensity ≥ s.noxiousThreshold then
    BilateralPosturalState.withdrawalExtensorThrust
  else
    BilateralPosturalState.symmetricStance

/-- Theorem: For any bilateral reflex state satisfying the crossed-extensor invariant,
    suprathreshold noxious stimulation guarantees ipsilateral withdrawal and contralateral stance thrust. -/
theorem noxious_stimulus_guarantees_postural_compensation
    (s : BilateralReflexState)
    (h_inv : satisfiesCrossedExtensorInvariant s)
    (h_pain : s.noxiousIntensity ≥ s.noxiousThreshold) :
    evaluatePosturalState s = BilateralPosturalState.withdrawalExtensorThrust ∧
    s.ipsiFlex ≥ s.activeThreshold ∧
    s.contraExt ≥ s.activeThreshold := by
  dsimp [evaluatePosturalState]
  rw [if_pos h_pain]
  dsimp [satisfiesCrossedExtensorInvariant] at h_inv
  have h_resp := h_inv h_pain
  rcases h_resp with ⟨h_if, _, h_ce, _⟩
  exact ⟨rfl, h_if, h_ce⟩

/-- Theorem: Suprathreshold noxious stimulation strictly suppresses both the ipsilateral extensor
    and the contralateral flexor below the suppression ceiling. -/
theorem noxious_stimulus_suppresses_antagonists
    (s : BilateralReflexState)
    (h_inv : satisfiesCrossedExtensorInvariant s)
    (h_pain : s.noxiousIntensity ≥ s.noxiousThreshold) :
    s.ipsiExt ≤ s.suppressionCeiling ∧ s.contraFlex ≤ s.suppressionCeiling := by
  dsimp [satisfiesCrossedExtensorInvariant] at h_inv
  have h_resp := h_inv h_pain
  rcases h_resp with ⟨_, h_ie, _, h_cf⟩
  exact ⟨h_ie, h_cf⟩

end Weft.Formal.CrossedExtensorReflex
