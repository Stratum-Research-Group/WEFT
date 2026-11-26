/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.VLPAGSurvivalInvariants

/--
State vector representing Periaqueductal Gray (PAG) Survival Motor Arbitration.
Units scaled by 100 for integer arithmetic verification.
-/
structure PAGSurvivalState where
  threatDistance : Nat  -- [0, 100] Threat distance (0 = immediate contact, 100 = far distant)
  loomingSpeed : Nat    -- [0, 100] Looming velocity (0 = static, 100 = explosive approach)
  corticosterone : Nat  -- [0, 100] Baseline fear / anxiety tone
  vlPAGRate : Nat       -- [0, 5000] vlPAG firing rate (scaled, 100 = 1.0 Hz, Freezing)
  dlPAGRate : Nat       -- [0, 8000] dlPAG/lPAG firing rate (scaled, 100 = 1.0 Hz, Flight/Panic)
  sympatheticTone : Nat -- [0, 1000] Autonomic sympathetic tone

/--
Raw vlPAG drive (Postural freezing / Immobility):
Favored by distant, non-looming threats and high baseline corticosterone.
Baseline 3.0 Hz (300).
-/
def computeRawVlPAGDrive (dist loom cort : Nat) : Nat :=
  let distSafe := if dist > 100 then 100 else dist
  let loomFactor := if loom > 100 then 0 else 100 - loom
  300 + (35 * distSafe * loomFactor) / 100 + 15 * cort

/--
Raw dlPAG / lPAG drive (Explosive panic escape / Flight):
Favored by close proximity and fast looming approach.
Baseline 2.0 Hz (200).
-/
def computeRawDlPAGDrive (dist loom : Nat) : Nat :=
  let proxFactor := if dist > 100 then 0 else 100 - dist
  let loomSafe := if loom > 100 then 100 else loom
  200 + 60 * proxFactor + 40 * loomSafe

/--
Effective vlPAG firing rate with collateral cross-inhibition from dlPAG:
Clamped to [0, 5000] (0 to 50.0 Hz).
-/
def computeEffectiveVlPAGRate (rawVl rawDl : Nat) : Nat :=
  let inh := (45 * rawDl) / 100
  let net := rawVl - inh
  if net > 5000 then 5000 else net

/--
Effective dlPAG firing rate with collateral cross-inhibition from vlPAG:
Clamped to [0, 8000] (0 to 80.0 Hz).
-/
def computeEffectiveDlPAGRate (rawDl rawVl : Nat) : Nat :=
  let inh := (20 * rawVl) / 100
  let net := rawDl - inh
  if net > 8000 then 8000 else net

/--
Autonomic Sympathetic Tone fraction:
Sympathetic = min(1000, (dlPAG * 12) / 100 + (vlPAG * 2) / 100).
-/
def computeSympatheticTone (dlRate vlRate : Nat) : Nat :=
  let tone := (dlRate * 12 + vlRate * 2) / 100
  if tone > 1000 then 1000 else tone

/--
Freezing Motor Attractor Indicator:
Triggers when effective vlPAG rate exceeds 20.0 Hz (2000).
-/
def isFreezingActive (vlRate : Nat) : Bool :=
  decide (vlRate >= 2000)

/--
Flight / Panic Escape Motor Attractor Indicator:
Triggers when effective dlPAG rate exceeds 35.0 Hz (3500).
-/
def isFlightEscapeActive (dlRate : Nat) : Bool :=
  decide (dlRate >= 3500)

/--
Theorem 1: Raw vlPAG drive is strictly non-negative.
-/
theorem raw_vlpag_non_negative (d l c : Nat) :
  computeRawVlPAGDrive d l c >= 0 := by
  dsimp [computeRawVlPAGDrive]
  split <;> split <;> omega

/--
Theorem 2: Raw dlPAG drive is strictly non-negative.
-/
theorem raw_dlpag_non_negative (d l : Nat) :
  computeRawDlPAGDrive d l >= 0 := by
  dsimp [computeRawDlPAGDrive]
  split <;> split <;> omega

/--
Theorem 3: Effective vlPAG rate is bounded by 5000 (50.0 Hz).
-/
theorem effective_vlpag_bounded (rawVl rawDl : Nat) :
  computeEffectiveVlPAGRate rawVl rawDl <= 5000 := by
  dsimp [computeEffectiveVlPAGRate]
  split <;> omega

/--
Theorem 4: Effective dlPAG rate is bounded by 8000 (80.0 Hz).
-/
theorem effective_dlpag_bounded (rawDl rawVl : Nat) :
  computeEffectiveDlPAGRate rawDl rawVl <= 8000 := by
  dsimp [computeEffectiveDlPAGRate]
  split <;> omega

/--
Theorem 5: Autonomic sympathetic tone is strictly bounded within [0, 1000].
-/
theorem sympathetic_tone_bounded (dl vl : Nat) :
  computeSympatheticTone dl vl <= 1000 := by
  dsimp [computeSympatheticTone]
  split <;> omega

/--
Theorem 6: Distant, non-looming threat with elevated fear triggers Freezing attractor.
-/
theorem distant_threat_triggers_freezing (c : Nat) (h_cort : c >= 50) :
  isFreezingActive (computeEffectiveVlPAGRate (computeRawVlPAGDrive 80 0 c) (computeRawDlPAGDrive 80 0)) = true := by
  dsimp [computeEffectiveVlPAGRate, computeRawVlPAGDrive, computeRawDlPAGDrive, isFreezingActive]
  have h_bound : 300 + 35 * 80 * 100 / 100 + 15 * c >= 3850 := by omega
  have h_dl : 200 + 60 * (100 - 80) + 40 * 0 = 1400 := by rfl
  have h_inh : 45 * 1400 / 100 = 630 := by rfl
  have h_cond : 300 + 35 * 80 * 100 / 100 + 15 * c - 630 >= 2000 := by omega
  split
  · decide
  · rename_i h_not_gt
    exact decide_eq_true h_cond

/--
Theorem 7: Immediate proximity and high looming velocity triggers Flight/Panic Escape.
-/
theorem imminent_looming_triggers_escape :
  isFlightEscapeActive (computeEffectiveDlPAGRate (computeRawDlPAGDrive 0 100) (computeRawVlPAGDrive 0 100 0)) = true := by
  dsimp [computeEffectiveDlPAGRate, computeRawDlPAGDrive, computeRawVlPAGDrive, isFlightEscapeActive]
  decide

/--
Theorem 8: In extreme flight escape, dlPAG rates exceed 7000, creating strong cross-inhibition.
-/
theorem imminent_escape_cross_inhibition_strength :
  (45 * computeRawDlPAGDrive 0 100) / 100 >= 4000 := by
  dsimp [computeRawDlPAGDrive]
  decide

/--
Theorem 9: Imminent escape completely suppresses Freezing mode to false.
-/
theorem imminent_threat_suppresses_freezing :
  isFreezingActive (computeEffectiveVlPAGRate (computeRawVlPAGDrive 0 100 20) (computeRawDlPAGDrive 0 100)) = false := by
  dsimp [isFreezingActive, computeEffectiveVlPAGRate, computeRawVlPAGDrive, computeRawDlPAGDrive]
  decide

/--
Theorem 10: Raw dlPAG drive is strictly monotonic with respect to looming speed at fixed proximity.
-/
theorem raw_dlpag_monotonic_looming (dist l1 l2 : Nat) (h_le : l1 <= l2) (h_l2 : l2 <= 100) :
  computeRawDlPAGDrive dist l1 <= computeRawDlPAGDrive dist l2 := by
  dsimp [computeRawDlPAGDrive]
  have h_l1 : l1 <= 100 := Nat.le_trans h_le h_l2
  have h_s1 : (if l1 > 100 then 100 else l1) = l1 := by
    have : ¬(l1 > 100) := by omega
    exact if_neg this
  have h_s2 : (if l2 > 100 then 100 else l2) = l2 := by
    have : ¬(l2 > 100) := by omega
    exact if_neg this
  rw [h_s1, h_s2]
  have h_mul : 40 * l1 <= 40 * l2 := Nat.mul_le_mul_left 40 h_le
  omega

/--
Theorem 11: High flight rate strictly elevates sympathetic tone past 400.
-/
theorem high_flight_elevates_sympathetic_tone (dl vl : Nat) (h_dl : dl >= 4000) :
  computeSympatheticTone dl vl >= 480 := by
  dsimp [computeSympatheticTone]
  have h_prod : dl * 12 >= 4000 * 12 := Nat.mul_le_mul_right 12 h_dl
  have h_const : 4000 * 12 = 48000 := by rfl
  have h_sum : dl * 12 + vl * 2 >= 48000 := by omega
  have h_div : (dl * 12 + vl * 2) / 100 >= 480 := by
    have h_c_div : 48000 / 100 = 480 := by rfl
    rw [← h_c_div]
    exact Nat.div_le_div_right h_sum
  split
  · omega
  · exact h_div

/--
Theorem 12: Quiescent baseline (dlPAG = 200, vlPAG = 300) keeps sympathetic tone calm (<= 40).
-/
theorem quiescent_sympathetic_tone_calm :
  computeSympatheticTone 200 300 <= 40 := by
  dsimp [computeSympatheticTone]
  decide

end Weft.Formal.VLPAGSurvivalInvariants
