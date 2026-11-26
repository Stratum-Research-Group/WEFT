-- Lateral Habenula (LHb) Anti-Reward Signaling, RMTg GABA Relay, and Dopamine Pause Invariants
-- Formal verification of habenular anti-reward coding and midbrain dopaminergic silencing:
-- 1. Matsumoto & Hikosaka (2007, 2009) Negative Reward Prediction Error (nRPE) and Aversion Coding.
-- 2. Proulx et al. (2014) & Jhou et al. (2009) Rostromedial Tegmental Nucleus (RMTg) GABAergic relay.
-- 3. Monotonic escalation of LHb firing by negative reward prediction errors and aversive stressors.
-- 4. Monotonic suppression of midbrain VTA dopamine firing by LHb-driven RMTg GABA release.
-- 5. Complete dopaminergic pause threshold during severe aversive burst firing.
-- Target #389 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.LateralHabenula

/-- State parameters representing the Lateral Habenula, RMTg, and VTA dopamine circuit.
    All firing rates, prediction errors, and stress levels are scaled natural units. -/
structure HabenulaState where
  /-- Expected reward value (scaled Nat, 0 - 100) -/
  expectedReward : Nat
  /-- Actual obtained reward outcome (scaled Nat, 0 - 100) -/
  actualOutcome : Nat
  /-- Afferent aversive stress / nociceptive input (scaled Nat, 0 - 100) -/
  aversiveStress : Nat
  /-- Baseline tonic LHb firing rate (scaled Nat, e.g. 8 representing 8 Hz) -/
  tonicBaseline : Nat
  /-- Maximum baseline dopamine firing rate in VTA (scaled Nat, e.g. 50 representing 5.0 Hz) -/
  daBaseline : Nat
  deriving Repr

/-- Negative Reward Prediction Error (nRPE):
    nRPE = if expectedReward > actualOutcome then expectedReward - actualOutcome else 0 -/
def negativeRPE (s : HabenulaState) : Nat :=
  if s.expectedReward > s.actualOutcome then s.expectedReward - s.actualOutcome else 0

/-- Lateral Habenula firing rate:
    Rate_LHb = tonicBaseline + (45 * negativeRPE) / 100 + (35 * aversiveStress) / 100 -/
def lhbFiringRate (s : HabenulaState) : Nat :=
  s.tonicBaseline + (45 * negativeRPE s) / 100 + (35 * s.aversiveStress) / 100

/-- RMTg GABAergic inhibitory output rate driving VTA dopamine neurons:
    Rate_RMTg = 5 + (85 * lhbFiringRate) / 100 -/
def rmtgGABAOutput (s : HabenulaState) : Nat :=
  5 + (85 * lhbFiringRate s) / 100

/-- VTA Dopamine firing rate suppressed by RMTg GABAergic transmission:
    DA_rate = if daBaseline >= rmtgGABAOutput then daBaseline - rmtgGABAOutput else 0 -/
def vtaDAFiringRate (s : HabenulaState) : Nat :=
  if s.daBaseline >= rmtgGABAOutput s then s.daBaseline - rmtgGABAOutput s else 0

/-- Theorem 1: LHb firing rate is strictly bounded below by tonic baseline. -/
theorem lhb_firing_bounded_by_baseline (s : HabenulaState) :
    lhbFiringRate s >= s.tonicBaseline := by
  dsimp [lhbFiringRate]
  omega

/-- Theorem 2: Reward omission (actualOutcome = 0, expectedReward > 0) strictly elevates nRPE above 0. -/
theorem reward_omission_positive_nrpe (s : HabenulaState) (h_exp : s.expectedReward > 0)
    (h_zero : s.actualOutcome = 0) :
    negativeRPE s > 0 := by
  dsimp [negativeRPE]
  rw [h_zero]
  split <;> omega

/-- Theorem 3: Increasing aversive stress monotonically elevates LHb firing rate. -/
theorem aversive_stress_monotone_lhb (s : HabenulaState) (stress1 stress2 : Nat)
    (h_stress : stress1 <= stress2) :
    lhbFiringRate { s with aversiveStress := stress1 } <=
    lhbFiringRate { s with aversiveStress := stress2 } := by
  dsimp [lhbFiringRate, negativeRPE]
  have h_mul : (35 * stress1) / 100 <= (35 * stress2) / 100 := by
    apply Nat.div_le_div_right
    omega
  omega

/-- Theorem 4: RMTg GABA output is monotonically increasing with LHb firing rate. -/
theorem lhb_activation_monotone_rmtg (s : HabenulaState) (extra_stress : Nat) :
    let s_stressed : HabenulaState := { s with aversiveStress := s.aversiveStress + extra_stress }
    rmtgGABAOutput s <= rmtgGABAOutput s_stressed := by
  dsimp [rmtgGABAOutput]
  have h_lhb_mono : lhbFiringRate s <= lhbFiringRate { s with aversiveStress := s.aversiveStress + extra_stress } := by
    apply aversive_stress_monotone_lhb
    omega
  have h_div : (85 * lhbFiringRate s) / 100 <= (85 * lhbFiringRate { s with aversiveStress := s.aversiveStress + extra_stress }) / 100 := by
    apply Nat.div_le_div_right
    omega
  omega

/-- Theorem 5: High LHb activation driving RMTg GABA beyond daBaseline triggers complete dopamine pause (DA = 0). -/
theorem dopamine_complete_pause_condition (s : HabenulaState)
    (h_suppress : rmtgGABAOutput s >= s.daBaseline) :
    vtaDAFiringRate s = 0 := by
  dsimp [vtaDAFiringRate]
  split <;> omega

end Weft.Formal.LateralHabenula
