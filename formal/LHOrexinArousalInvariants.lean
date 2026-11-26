/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.LHOrexinArousalInvariants

/--
State vector representing Lateral Hypothalamus (LH) Orexin Arousal & Muscle Tone Gating.
Units scaled by 100 for integer arithmetic verification.
-/
structure OrexinState where
  emotionalValence : Nat  -- [0, 100] Positive emotional / reward arousal surge
  orexinIntegrity : Nat   -- [0, 100] Surviving orexin neuron fraction (100 = intact, 0 = narcoleptic)
  circadianDrive : Nat    -- [0, 100] Circadian wake promotion drive
  orexinRate : Nat        -- [0, 4500] LH Orexin firing rate (scaled, 100 = 1.0 Hz)
  monoaminergicTone : Nat -- [0, 1000] LC / DR motoneuron facilitation
  sldAtoniaDrive : Nat    -- [0, 1000] Sublaterodorsal nucleus motor atonia pathway
  somaticMuscleTone : Nat -- [0, 1000] Somatic postural tone

/--
LH Orexin firing rate:
Baseline 3.0 Hz (300) + 25 * emotionalValence + 12 * circadianDrive, gated by orexinIntegrity.
Clamped to [0, 4500] (0 to 45.0 Hz).
-/
def computeOrexinRate (valence integrity circ : Nat) : Nat :=
  let rawDrive := 300 + 25 * valence + 12 * circ
  let gated := (rawDrive * integrity) / 100
  if gated > 4500 then 4500 else gated

/--
Monoaminergic motoneuron facilitation tone (LC Noradrenaline + DR Serotonin):
Baseline 150 (15.0%) + 2 * orexinRate.
Clamped to [0, 1000] (100.0%).
-/
def computeMonoaminergicTone (orexinRate : Nat) : Nat :=
  let tone := 150 + (2 * orexinRate) / 10
  if tone > 1000 then 1000 else tone

/--
Sublaterodorsal (SLD) nucleus atonia / paralysis drive:
Emotional valence excites SLD (9 * valence), but is suppressed by monoamines ((75 * mono) / 100).
Clamped to [0, 1000].
-/
def computeSLDAtoniaDrive (valence monoTone : Nat) : Nat :=
  let rawExcit := 9 * valence
  let inh := (75 * monoTone) / 100
  let net := rawExcit - inh
  if net > 1000 then 1000 else net

/--
Somatic Postural Muscle Tone:
Tone = ((90 * mono) / 100) - ((110 * sldAtonia) / 100).
Clamped to [0, 1000].
-/
def computeMuscleTone (monoTone sldAtonia : Nat) : Nat :=
  let drive := (90 * monoTone) / 100
  let atonia := (110 * sldAtonia) / 100
  let net := drive - atonia
  if net > 1000 then 1000 else net

/--
Cataplexy Collapse Indicator:
Triggers when SLD atonia drive exceeds 500 (50.0%) and muscle tone falls below 200 (20.0%).
-/
def isCataplexyCollapse (sldAtonia muscleTone : Nat) : Bool :=
  decide (sldAtonia >= 500 ∧ muscleTone < 200)

/--
Theorem 1: LH Orexin firing rate is non-negative.
-/
theorem orexin_rate_non_negative (v i c : Nat) :
  computeOrexinRate v i c >= 0 := by
  dsimp [computeOrexinRate]
  split <;> omega

/--
Theorem 2: LH Orexin firing rate is upper bounded by 4500 (45.0 Hz).
-/
theorem orexin_rate_upper_bound (v i c : Nat) :
  computeOrexinRate v i c <= 4500 := by
  dsimp [computeOrexinRate]
  split <;> omega

/--
Theorem 3: Monoaminergic tone is strictly bounded within [0, 1000].
-/
theorem monoaminergic_tone_bounded (rate : Nat) :
  computeMonoaminergicTone rate <= 1000 := by
  dsimp [computeMonoaminergicTone]
  split <;> omega

/--
Theorem 4: SLD atonia drive is strictly bounded within [0, 1000].
-/
theorem sld_atonia_drive_bounded (v m : Nat) :
  computeSLDAtoniaDrive v m <= 1000 := by
  dsimp [computeSLDAtoniaDrive]
  split <;> omega

/--
Theorem 5: Somatic muscle tone is strictly bounded within [0, 1000].
-/
theorem muscle_tone_bounded (m a : Nat) :
  computeMuscleTone m a <= 1000 := by
  dsimp [computeMuscleTone]
  split <;> omega

/--
Theorem 6: Orexin firing rate is monotonic with respect to emotional valence at fixed integrity.
-/
theorem orexin_rate_monotonic_valence (v1 v2 i c : Nat) (h_le : v1 <= v2) :
  computeOrexinRate v1 i c <= computeOrexinRate v2 i c := by
  dsimp [computeOrexinRate]
  have h_drive : 300 + 25 * v1 + 12 * c <= 300 + 25 * v2 + 12 * c := by omega
  have h_prod : (300 + 25 * v1 + 12 * c) * i <= (300 + 25 * v2 + 12 * c) * i :=
    Nat.mul_le_mul_right i h_drive
  have h_div : ((300 + 25 * v1 + 12 * c) * i) / 100 <= ((300 + 25 * v2 + 12 * c) * i) / 100 :=
    Nat.div_le_div_right h_prod
  split
  · split <;> omega
  · split <;> omega

/--
Theorem 7: Low SLD atonia (atonia = 0) strictly prevents cataplexy.
-/
theorem zero_atonia_prevents_cataplexy (tone : Nat) :
  isCataplexyCollapse 0 tone = false := by
  dsimp [isCataplexyCollapse]
  have h_not : ¬(0 >= 500 ∧ tone < 200) := by omega
  exact decide_eq_false h_not

/--
Theorem 8: Total orexin neuron degeneration (integrity = 0) eliminates orexin firing.
-/
theorem narcoleptic_orexin_loss_zero_rate (v c : Nat) :
  computeOrexinRate v 0 c = 0 := by
  have h_zero : (300 + 25 * v + 12 * c) * 0 / 100 = 0 := by omega
  have h_not : ¬((300 + 25 * v + 12 * c) * 0 / 100 > 4500) := by omega
  change (if (300 + 25 * v + 12 * c) * 0 / 100 > 4500 then 4500 else (300 + 25 * v + 12 * c) * 0 / 100) = 0
  rw [if_neg h_not]
  exact h_zero

/--
Theorem 9: In narcoleptic state (orexin = 0), emotional surge triggers cataplectic collapse.
-/
theorem narcoleptic_emotional_surge_triggers_cataplexy :
  isCataplexyCollapse (computeSLDAtoniaDrive 90 (computeMonoaminergicTone 0))
                      (computeMuscleTone (computeMonoaminergicTone 0)
                                         (computeSLDAtoniaDrive 90 (computeMonoaminergicTone 0))) = true := by
  dsimp [computeMonoaminergicTone, computeSLDAtoniaDrive, computeMuscleTone, isCataplexyCollapse]
  decide

/--
Theorem 10: Monoaminergic tone is strictly monotonic with respect to orexin firing rate.
-/
theorem monoaminergic_tone_monotonic (r1 r2 : Nat) (h_le : r1 <= r2) :
  computeMonoaminergicTone r1 <= computeMonoaminergicTone r2 := by
  dsimp [computeMonoaminergicTone]
  have h_mul : 2 * r1 <= 2 * r2 := Nat.mul_le_mul_left 2 h_le
  have h_div : (2 * r1) / 10 <= (2 * r2) / 10 := Nat.div_le_div_right h_mul
  split
  · split <;> omega
  · split <;> omega

/--
Theorem 11: High orexin rate (>= 3000) drives monoaminergic facilitation past 700.
-/
theorem high_orexin_drives_monoamines (r : Nat) (h_ge : r >= 3000) :
  computeMonoaminergicTone r >= 750 := by
  dsimp [computeMonoaminergicTone]
  have h_div : (2 * r) / 10 >= 600 := by
    have h_prod : 2 * r >= 6000 := by omega
    have h_c : 6000 / 10 = 600 := by rfl
    rw [← h_c]
    exact Nat.div_le_div_right h_prod
  split
  · omega
  · omega

/--
Theorem 12: Zero emotional valence yields zero SLD atonia drive.
-/
theorem zero_valence_zero_atonia (m : Nat) :
  computeSLDAtoniaDrive 0 m = 0 := by
  dsimp [computeSLDAtoniaDrive]
  have h_zero : 9 * 0 - (75 * m) / 100 = 0 := by omega
  split
  · omega
  · omega

end Weft.Formal.LHOrexinArousalInvariants
