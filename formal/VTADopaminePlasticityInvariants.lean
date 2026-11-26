-- Ventral Tegmental Area (VTA) Dopaminergic Plasticity & Three-Factor Synaptic Invariants
-- Formal verification of dopamine-gated three-factor learning rules and eligibility trace dynamics:
-- 1. Reynolds & Wickens (2002) & Gerstner et al. (2018) Three-Factor Synaptic Plasticity.
-- 2. Factor 1 (Presynaptic) x Factor 2 (Postsynaptic) non-negative eligibility trace tagging.
-- 3. Factor 3 (Extracellular Dopamine RPE) sign-dependent consolidation:
--    - Positive RPE (> baseline): Long-Term Potentiation (LTP).
--    - Negative RPE (< baseline): Long-Term Depression (LTD).
-- 4. Bounded synaptic conductance invariants [W_min, W_max].
-- Target #390 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.VTADopaminePlasticity

/-- State parameters representing pre/post synaptic firing, eligibility trace, and dopamine signal.
    All rates, weights, and concentrations are scaled natural units. -/
structure PlasticityState where
  /-- Presynaptic spike rate (scaled Nat, 0 - 100) -/
  preRate : Nat
  /-- Postsynaptic spike rate / dendritic depolarization (scaled Nat, 0 - 100) -/
  postRate : Nat
  /-- Ambient dopamine concentration (scaled Nat, 0 - 100, where baseline = 50) -/
  dopamineLevel : Nat
  /-- Current synaptic weight (scaled Nat, e.g. 100 representing 1.0) -/
  currentWeight : Nat
  /-- Minimum allowable weight (scaled Nat, e.g. 10) -/
  minWeight : Nat
  /-- Maximum allowable weight (scaled Nat, e.g. 350) -/
  maxWeight : Nat
  h_bounds : minWeight <= maxWeight
  deriving Repr

/-- Baseline ambient dopamine level separating positive from negative RPE -/
def daBaseline : Nat := 50

/-- Factor 1 x Factor 2: Pre-post Hebbian coincidence eligibility trace:
    e = (preRate * postRate) / 100 -/
def eligibilityTrace (s : PlasticityState) : Nat :=
  (s.preRate * s.postRate) / 100

/-- Positive Reward Prediction Error magnitude:
    RPE+ = if dopamineLevel > daBaseline then dopamineLevel - daBaseline else 0 -/
def positiveRPE (s : PlasticityState) : Nat :=
  if s.dopamineLevel > daBaseline then s.dopamineLevel - daBaseline else 0

/-- Negative Reward Prediction Error magnitude:
    RPE- = if dopamineLevel < daBaseline then daBaseline - s.dopamineLevel else 0 -/
def negativeRPE (s : PlasticityState) : Nat :=
  if s.dopamineLevel < daBaseline then daBaseline - s.dopamineLevel else 0

/-- Raw potentiation increment (LTP delta):
    dW+ = (eligibilityTrace * positiveRPE) / 100 -/
def ltpIncrement (s : PlasticityState) : Nat :=
  (eligibilityTrace s * positiveRPE s) / 100

/-- Raw depression decrement (LTD delta):
    dW- = (eligibilityTrace * negativeRPE) / 100 -/
def ltdDecrement (s : PlasticityState) : Nat :=
  (eligibilityTrace s * negativeRPE s) / 100

/-- Net intermediate decayed weight:
    decayed = if W + dW+ >= dW- then W + dW+ - dW- else 0 -/
def intermediateWeight (s : PlasticityState) : Nat :=
  let raw := s.currentWeight + ltpIncrement s
  if raw >= ltdDecrement s then raw - ltdDecrement s else 0

/-- Updated synaptic weight applying bounded three-factor learning:
    Clamp intermediate weight to [minWeight, maxWeight] -/
def updatedWeight (s : PlasticityState) : Nat :=
  let decayed := intermediateWeight s
  if decayed <= s.minWeight then s.minWeight
  else if decayed >= s.maxWeight then s.maxWeight
  else decayed

/-- Theorem 1: Eligibility trace is monotonic with presynaptic activity. -/
theorem eligibility_monotone_pre (s : PlasticityState) (pre1 pre2 : Nat) (h_pre : pre1 <= pre2) :
    eligibilityTrace { s with preRate := pre1 } <=
    eligibilityTrace { s with preRate := pre2 } := by
  dsimp [eligibilityTrace]
  apply Nat.div_le_div_right
  apply Nat.mul_le_mul_right
  exact h_pre

/-- Theorem 2: In the absence of presynaptic or postsynaptic activity, eligibility trace is 0. -/
theorem zero_activity_zero_trace (s : PlasticityState) (h_zero : s.preRate = 0) :
    eligibilityTrace s = 0 := by
  dsimp [eligibilityTrace]
  rw [h_zero]
  omega

/-- Theorem 3: When dopamine is at baseline (RPE = 0), both LTP and LTD increments vanish. -/
theorem baseline_dopamine_neutral_plasticity (s : PlasticityState) (h_base : s.dopamineLevel = daBaseline) :
    ltpIncrement s = 0 /\ ltdDecrement s = 0 := by
  dsimp [ltpIncrement, ltdDecrement, positiveRPE, negativeRPE]
  rw [h_base]
  dsimp [daBaseline]
  constructor <;> omega

/-- Theorem 4: Positive dopamine burst (RPE > 0) produces zero LTD decrement. -/
theorem positive_dopamine_zero_ltd (s : PlasticityState) (h_pos : s.dopamineLevel > daBaseline) :
    ltdDecrement s = 0 := by
  dsimp [ltdDecrement, negativeRPE]
  split
  · omega
  · omega

/-- Theorem 5: Synaptic weight is strictly invariant-bounded within [minWeight, maxWeight]. -/
theorem updated_weight_bounded (s : PlasticityState) :
    s.minWeight <= updatedWeight s /\ updatedWeight s <= s.maxWeight := by
  dsimp [updatedWeight]
  have h_bound := s.h_bounds
  split
  · omega
  · split <;> omega

end Weft.Formal.VTADopaminePlasticity
