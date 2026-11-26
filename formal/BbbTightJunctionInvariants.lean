/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.BbbTightJunction

/-
Target #407: Blood-Brain Barrier (BBB) Endothelial Tight-Junction Restoration & Pericyte Tie2 Signaling Invariants
Pericyte-endothelial crosstalk via Angiopoietin-1/Tie2 receptor activation,
TIMP-1/2 suppression of matrix metalloproteinases (MMP-2/9), claudin-5 and occludin
junctional assembly, trans-endothelial electrical resistance (TEER), and paracellular leakage prevention.
-/

/-- Phosphorylated Tie2 receptor activation percentage (0 - 100)
    tie2 = (200 + pericyte_pct * 3 + ang1_x10 * 32) * (200 - ros_pct) / 2000
-/
def compute_tie2_activation (pericyte_pct : Nat) (ang1_x10 : Nat) (ros_pct : Nat) : Nat :=
  let raw := 200 + pericyte_pct * 3 + ang1_x10 * 32
  let ros_mod := if ros_pct > 200 then 0 else 200 - ros_pct
  let eff := (raw * ros_mod) / 2000
  if eff > 100 then 100 else eff

/-- Effective MMP-2/9 matrix metalloproteinase activity scaled x10 -/
def compute_effective_mmp (mmp_x10 : Nat) (shed_dose_pct : Nat) : Nat :=
  let suppression := (shed_dose_pct * 75) / 100
  let mod_factor := if suppression > 80 then 20 else 100 - suppression
  (mmp_x10 * mod_factor) / 100

/-- Claudin-5 expression percentage (0 - 100) -/
def compute_claudin5_expression (tie2_pct : Nat) (eff_mmp_x10 : Nat) : Nat :=
  let positive := (tie2_pct * 11) / 10
  let degradation := (eff_mmp_x10 * 12) / 10
  if positive > degradation then
    let raw := positive - degradation
    if raw > 100 then 100 else (if raw < 5 then 5 else raw)
  else 5

/-- Occludin expression percentage (0 - 100) -/
def compute_occludin_expression (tie2_pct : Nat) (eff_mmp_x10 : Nat) : Nat :=
  let positive := (tie2_pct * 105) / 100
  let degradation := (eff_mmp_x10 * 105) / 100
  if positive > degradation then
    let raw := positive - degradation
    if raw > 100 then 100 else (if raw < 5 then 5 else raw)
  else 5

/-- Trans-endothelial electrical resistance (TEER) in Ohm*cm^2 -/
def compute_teer_resistance (claudin_pct : Nat) (occludin_pct : Nat) : Nat :=
  200 + (claudin_pct * 85) / 10 + (occludin_pct * 45) / 10

/-- Paracellular permeability leakage percentage (0 - 100) -/
def compute_paracellular_leakage (teer : Nat) : Nat :=
  if teer ≥ 1500 then 2
  else 100 - (teer * 100) / 1500

theorem severe_vascular_injury_destroys_bbb_integrity :
  let tie2 := compute_tie2_activation 22 0 95
  let eff_mmp := compute_effective_mmp 42 0
  let claudin := compute_claudin5_expression tie2 eff_mmp
  let occludin := compute_occludin_expression tie2 eff_mmp
  let teer := compute_teer_resistance claudin occludin
  let leakage := compute_paracellular_leakage teer
  teer ≤ 400 ∧ leakage ≥ 70 := by
  dsimp [compute_tie2_activation, compute_effective_mmp, compute_claudin5_expression,
         compute_occludin_expression, compute_teer_resistance, compute_paracellular_leakage]
  decide

theorem shed_cm_infusion_fully_reconstitutes_bbb_barrier :
  -- Under 100% SHED-CM infusion (Ang-1 = 18 ng/mL -> ang1_x10 = 180, pericyte = 85%, ROS = 20%)
  let tie2 := compute_tie2_activation 85 180 20
  let eff_mmp := compute_effective_mmp 15 100
  let claudin := compute_claudin5_expression tie2 eff_mmp
  let occludin := compute_occludin_expression tie2 eff_mmp
  let teer := compute_teer_resistance claudin occludin
  let leakage := compute_paracellular_leakage teer
  teer ≥ 1200 ∧ leakage ≤ 15 := by
  dsimp [compute_tie2_activation, compute_effective_mmp, compute_claudin5_expression,
         compute_occludin_expression, compute_teer_resistance, compute_paracellular_leakage]
  decide

end Weft.Formal.BbbTightJunction
