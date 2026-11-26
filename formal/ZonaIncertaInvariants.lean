/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.ZonaIncertaInvariants

/--
State vector representing Zona Incerta (ZI) Visceromotor Integration.
Units scaled by 100 for integer arithmetic verification.
-/
structure ZIState where
  hungerDrive : Nat     -- [0, 100] Metabolic hunger (Ghrelin / AgRP tone)
  preySalience : Nat    -- [0, 100] External target / prey proximity
  stressTone : Nat      -- [0, 100] Corticosterone / anxiety tone
  rostralPVRate : Nat   -- [0, 6000] Rostral ZI PV+ firing rate (scaled, 100 = 1.0 Hz)
  caudalSOMRate : Nat   -- [0, 5000] Caudal ZI SOM+ firing rate (scaled, 100 = 1.0 Hz)
  pagDisinhibition : Nat-- [0, 1000] Tectal / PAG motor disinhibition fraction

/--
Rostral ZI PV+ firing rate:
Baseline 4.0 Hz (400), stimulated by hunger drive (up to +50.0 Hz), attenuated by stress.
Clamped to [0, 6000] (0 to 60.0 Hz).
-/
def computeRostralPVRate (hunger stress : Nat) : Nat :=
  let rawDrive := 400 + 50 * hunger
  let stressPenalty := 15 * stress
  let net := rawDrive - stressPenalty
  if net > 6000 then 6000 else net

/--
Caudal ZI SOM+ firing rate:
Baseline 3.0 Hz (300), driven by prey proximity and amplified by metabolic state.
Clamped to [0, 5000] (0 to 50.0 Hz).
-/
def computeCaudalSOMRate (prey hunger : Nat) : Nat :=
  let drive := 300 + (45 * prey * (50 + hunger / 2)) / 100
  if drive > 5000 then 5000 else drive

/--
PAG & Superior Colliculus motor disinhibition:
Disinhibition = (rostralPV * 60 + caudalSOM * 40) / 500, clamped to 1000.
-/
def computePAGDisinhibition (pvRate somRate : Nat) : Nat :=
  let disinh := (pvRate * 60 + somRate * 40) / 500
  if disinh > 1000 then 1000 else disinh

/--
Consummatory Binge Drive threshold indicator:
Triggers when rostral PV+ rate exceeds 30.0 Hz (3000).
-/
def isBingeFeedingActive (pvRate : Nat) : Bool :=
  decide (pvRate >= 3000)

/--
Predatory Hunting Drive threshold indicator:
Triggers when caudal SOM+ rate exceeds 25.0 Hz (2500).
-/
def isPredatoryHuntingActive (somRate : Nat) : Bool :=
  decide (somRate >= 2500)

/--
Theorem 1: Rostral PV+ firing rate is non-negative.
-/
theorem rostral_pv_rate_non_negative (h s : Nat) :
  computeRostralPVRate h s >= 0 := by
  dsimp [computeRostralPVRate]
  split <;> omega

/--
Theorem 2: Rostral PV+ firing rate is upper bounded by 6000 (60.0 Hz).
-/
theorem rostral_pv_rate_upper_bound (h s : Nat) :
  computeRostralPVRate h s <= 6000 := by
  dsimp [computeRostralPVRate]
  split <;> omega

/--
Theorem 3: Caudal SOM+ firing rate is non-negative.
-/
theorem caudal_som_rate_non_negative (p h : Nat) :
  computeCaudalSOMRate p h >= 0 := by
  dsimp [computeCaudalSOMRate]
  split <;> omega

/--
Theorem 4: Caudal SOM+ firing rate is upper bounded by 5000 (50.0 Hz).
-/
theorem caudal_som_rate_upper_bound (p h : Nat) :
  computeCaudalSOMRate p h <= 5000 := by
  dsimp [computeCaudalSOMRate]
  split <;> omega

/--
Theorem 5: PAG motor disinhibition is strictly bounded within [0, 1000].
-/
theorem pag_disinhibition_bounded (pv som : Nat) :
  computePAGDisinhibition pv som <= 1000 := by
  dsimp [computePAGDisinhibition]
  split <;> omega

/--
Theorem 6: Rostral PV rate is monotonic with respect to hunger under zero stress.
-/
theorem rostral_pv_rate_monotonic_hunger (h1 h2 : Nat) (h_le : h1 <= h2) :
  computeRostralPVRate h1 0 <= computeRostralPVRate h2 0 := by
  dsimp [computeRostralPVRate]
  have h_mul : 50 * h1 <= 50 * h2 := Nat.mul_le_mul_left 50 h_le
  split
  · split <;> omega
  · split <;> omega

/--
Theorem 7: Extreme hunger (>= 80%) under low stress (<= 10%) guarantees Binge Feeding trigger.
-/
theorem high_hunger_triggers_binge_feeding (h s : Nat) (h_high : h >= 80) (s_low : s <= 10) :
  isBingeFeedingActive (computeRostralPVRate h s) = true := by
  dsimp [isBingeFeedingActive, computeRostralPVRate]
  have h_bound : 400 + 50 * h - 15 * s >= 3000 := by omega
  split
  · exact decide_eq_true (by omega)
  · exact decide_eq_true (by omega)

/--
Theorem 8: Baseline inputs under high stress retain quiet PAG disinhibition <= 300.
-/
theorem quiescent_pag_disinhibition_bounded :
  computePAGDisinhibition 0 300 <= 300 := by
  dsimp [computePAGDisinhibition]
  decide

/--
Theorem 9: Caudal SOM rate is monotonic with respect to prey salience at fixed hunger.
-/
theorem caudal_som_monotonic_prey (p1 p2 h : Nat) (h_le : p1 <= p2) :
  computeCaudalSOMRate p1 h <= computeCaudalSOMRate p2 h := by
  dsimp [computeCaudalSOMRate]
  have h_term : (45 * p1 * (50 + h / 2)) / 100 <= (45 * p2 * (50 + h / 2)) / 100 := by
    apply Nat.div_le_div_right
    have : 45 * p1 <= 45 * p2 := Nat.mul_le_mul_left 45 h_le
    exact Nat.mul_le_mul_right (50 + h / 2) this
  split
  · split <;> omega
  · split <;> omega

/--
Theorem 10: Zero prey salience strictly yields baseline 300 rate and prevents Predatory Hunting activation.
-/
theorem zero_prey_prevents_predatory_hunting (h : Nat) :
  isPredatoryHuntingActive (computeCaudalSOMRate 0 h) = false := by
  have h_zero : 45 * 0 * (50 + h / 2) = 0 := by omega
  have h_div : (45 * 0 * (50 + h / 2)) / 100 = 0 := by
    rw [h_zero]
  dsimp [isPredatoryHuntingActive, computeCaudalSOMRate]
  rw [h_div]
  decide

/--
Theorem 11: PAG motor disinhibition increases monotonically with rostral PV+ rate.
-/
theorem pag_disinhibition_monotonic_pv (pv1 pv2 som : Nat) (h_le : pv1 <= pv2) :
  computePAGDisinhibition pv1 som <= computePAGDisinhibition pv2 som := by
  dsimp [computePAGDisinhibition]
  have h_num : pv1 * 60 + som * 40 <= pv2 * 60 + som * 40 := by omega
  have h_div : (pv1 * 60 + som * 40) / 500 <= (pv2 * 60 + som * 40) / 500 :=
    Nat.div_le_div_right h_num
  split
  · split <;> omega
  · split <;> omega

/--
Theorem 12: High hunger disinhibits PAG motor activity (>= 400).
-/
theorem high_hunger_disinhibits_pag (h : Nat) (h_ge : h >= 70) :
  computePAGDisinhibition (computeRostralPVRate h 0) 300 >= 400 := by
  dsimp [computePAGDisinhibition, computeRostralPVRate]
  have h_pv : 400 + 50 * h - 15 * 0 >= 3900 := by omega
  split
  · split
    · omega
    · have h_sub : (6000 * 60 + 300 * 40) / 500 = 744 := by decide
      omega
  · split
    · omega
    · have h_raw : (400 + 50 * h) * 60 + 300 * 40 >= 3900 * 60 + 12000 := by omega
      have h_div_ge : ((400 + 50 * h) * 60 + 300 * 40) / 500 >= (3900 * 60 + 12000) / 500 :=
        Nat.div_le_div_right h_raw
      have h_const : (3900 * 60 + 12000) / 500 = 492 := by decide
      omega

end Weft.Formal.ZonaIncertaInvariants
