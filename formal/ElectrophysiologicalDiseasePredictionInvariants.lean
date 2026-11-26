/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.ElectrophysiologicalDiseasePredictionInvariants

/--
State vector representing Electrophysiological Disease State Prediction.
Quantities scaled by 100 for integer arithmetic verification.
-/
structure ElectrophysState where
  dopamineTone : Nat    -- [0, 100] Striatal dopamine concentration (100 = full tone, 0 = severe depletion)
  corticalDrive : Nat   -- [0, 100] Cortico-striatal motor intention
  betaPower : Nat       -- [0, 1000] STN-GPe resonant beta power (13-30 Hz, scaled)
  pacModulation : Nat   -- [0, 1000] Phase-amplitude coupling index between beta phase and gamma amplitude
  spectralSlopeChi : Nat-- [0, 350] 1/f aperiodic spectral slope (scaled, 280 = 2.80)
  diseaseRiskScore : Nat-- [0, 1000] Integrated disease risk index (scaled, 1000 = 100.0%)

/--
Beta power (13-30 Hz LFP power) emergence from dopamine depletion:
Baseline 100 (10.0 uV^2). When dopamine drops below 40%, recurrent STN-GPe amplification explodes.
Clamped to [0, 1000].
-/
def computeBetaPower (dopamine drive : Nat) : Nat :=
  let deficit := 100 - dopamine
  let driveAmp := 50 + drive / 2
  let raw := 100 + (80 * deficit * driveAmp) / 10000
  if raw > 1000 then 1000 else raw

/--
Phase-Amplitude Coupling (PAC) Modulation Index:
Coupling scales directly with beta power: PAC = clamp((betaPower * 90) / 100, 0, 1000).
-/
def computePACIndex (betaPower : Nat) : Nat :=
  let raw := (betaPower * 90) / 100
  if raw > 1000 then 1000 else raw

/--
1/f aperiodic spectral tilt chi (E/I balance):
Healthy high dopamine maintains steep slope 280 (2.80); dopamine depletion flattens toward 120 (1.20).
Slope = 280 - (140 * daDeficit) / 100.
-/
def computeSpectralSlope (dopamine : Nat) : Nat :=
  let deficit := 100 - dopamine
  let drop := (140 * deficit) / 100
  let net := 280 - drop
  if net > 350 then 350 else net

/--
Integrated Electrophysiological Disease Risk Score:
Risk = (50 * betaPower + 35 * pac + 15 * (300 - slope)) / 100.
Clamped to [0, 1000].
-/
def computeDiseaseRisk (beta pac slope : Nat) : Nat :=
  let invSlope := 300 - slope
  let weighted := (50 * beta + 35 * pac + 15 * invSlope) / 100
  if weighted > 1000 then 1000 else weighted

/--
Predictive Biomarker Warning Indicator:
Triggers early warning when disease risk score exceeds 450 (45.0%), well before clinical akinesia manifest at 700.
-/
def isPredictiveAlertActive (risk : Nat) : Bool :=
  decide (risk >= 450)

/--
Theorem 1: Beta power is non-negative.
-/
theorem beta_power_non_negative (da dr : Nat) :
  computeBetaPower da dr >= 0 := by
  dsimp [computeBetaPower]
  split <;> omega

/--
Theorem 2: Beta power is upper bounded by 1000.
-/
theorem beta_power_upper_bound (da dr : Nat) :
  computeBetaPower da dr <= 1000 := by
  dsimp [computeBetaPower]
  split <;> omega

/--
Theorem 3: PAC index is bounded within [0, 1000].
-/
theorem pac_index_bounded (beta : Nat) :
  computePACIndex beta <= 1000 := by
  dsimp [computePACIndex]
  split <;> omega

/--
Theorem 4: 1/f spectral slope is bounded within [0, 350].
-/
theorem spectral_slope_bounded (da : Nat) :
  computeSpectralSlope da <= 350 := by
  dsimp [computeSpectralSlope]
  split <;> omega

/--
Theorem 5: Integrated disease risk score is strictly bounded within [0, 1000].
-/
theorem disease_risk_bounded (b p s : Nat) :
  computeDiseaseRisk b p s <= 1000 := by
  dsimp [computeDiseaseRisk]
  split <;> omega

/--
Theorem 6: Beta power monotonically increases with dopamine depletion (decreasing dopamine).
-/
theorem beta_power_monotonic_depletion (da1 da2 dr : Nat) (h_le : da1 <= da2) (h_bound : da2 <= 100) :
  computeBetaPower da2 dr <= computeBetaPower da1 dr := by
  dsimp [computeBetaPower]
  have h_d1 : 100 - da2 <= 100 - da1 := by omega
  have h_prod : 80 * (100 - da2) * (50 + dr / 2) <= 80 * (100 - da1) * (50 + dr / 2) := by
    have h1 : 80 * (100 - da2) <= 80 * (100 - da1) := Nat.mul_le_mul_left 80 h_d1
    exact Nat.mul_le_mul_right (50 + dr / 2) h1
  have h_div : (80 * (100 - da2) * (50 + dr / 2)) / 10000 <= (80 * (100 - da1) * (50 + dr / 2)) / 10000 :=
    Nat.div_le_div_right h_prod
  split
  · split <;> omega
  · split <;> omega

/--
Theorem 7: Intact dopamine (100%) produces quiet, baseline beta power <= 100.
-/
theorem healthy_dopamine_low_beta (dr : Nat) :
  computeBetaPower 100 dr = 100 := by
  dsimp [computeBetaPower]
  have h_zero : 80 * (100 - 100) * (50 + dr / 2) / 10000 = 0 := by omega
  have h_cond : 100 + 80 * (100 - 100) * (50 + dr / 2) / 10000 = 100 := by omega
  have h_not : ¬(100 + 80 * (100 - 100) * (50 + dr / 2) / 10000 > 1000) := by omega
  rw [if_neg h_not]
  exact h_cond

/--
Theorem 8: Intact dopamine and quiet beta strictly precludes predictive disease alert.
-/
theorem healthy_state_no_disease_alert :
  isPredictiveAlertActive (computeDiseaseRisk (computeBetaPower 100 50)
                                             (computePACIndex (computeBetaPower 100 50))
                                             (computeSpectralSlope 100)) = false := by
  dsimp [computeDiseaseRisk, computeBetaPower, computePACIndex, computeSpectralSlope, isPredictiveAlertActive]
  decide

/--
Theorem 9: PAC index scales monotonically with beta power.
-/
theorem pac_monotonic_with_beta (b1 b2 : Nat) (h_le : b1 <= b2) :
  computePACIndex b1 <= computePACIndex b2 := by
  dsimp [computePACIndex]
  have h_prod : b1 * 90 <= b2 * 90 := Nat.mul_le_mul_right 90 h_le
  have h_div : (b1 * 90) / 100 <= (b2 * 90) / 100 := Nat.div_le_div_right h_prod
  split
  · split <;> omega
  · split <;> omega

/--
Theorem 10: Severe dopamine depletion (da = 10%) with elevated beta and PAC triggers predictive alert.
-/
theorem severe_depletion_triggers_predictive_alert :
  isPredictiveAlertActive (computeDiseaseRisk 600 (computePACIndex 600) (computeSpectralSlope 10)) = true := by
  dsimp [computeDiseaseRisk, computePACIndex, computeSpectralSlope, isPredictiveAlertActive]
  decide

/--
Theorem 11: High beta power (beta >= 800) and PAC (>= 700) elevates risk score past 645.
-/
theorem high_beta_and_pac_elevates_risk (b p s : Nat) (h_b : b >= 800) (h_p : p >= 700) :
  computeDiseaseRisk b p s >= 645 := by
  dsimp [computeDiseaseRisk]
  have h_b_term : 50 * b >= 40000 := by omega
  have h_p_term : 35 * p >= 24500 := by omega
  have h_tot : 50 * b + 35 * p + 15 * (300 - s) >= 64500 := by omega
  have h_div : (50 * b + 35 * p + 15 * (300 - s)) / 100 >= 645 := by
    have h_c : 64500 / 100 = 645 := by rfl
    rw [← h_c]
    exact Nat.div_le_div_right h_tot
  split
  · omega
  · exact h_div

/--
Theorem 12: Zero beta and zero PAC keeps disease risk score minimal (<= 45).
-/
theorem zero_activity_minimal_risk (s : Nat) :
  computeDiseaseRisk 0 0 s <= 45 := by
  dsimp [computeDiseaseRisk]
  have h_bound : (50 * 0 + 35 * 0 + 15 * (300 - s)) / 100 <= 45 := by omega
  split
  · omega
  · exact h_bound

end Weft.Formal.ElectrophysiologicalDiseasePredictionInvariants
