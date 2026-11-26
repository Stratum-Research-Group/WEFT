/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.GlymphaticAqp4

/-
Target #410: Glymphatic Convective Flux & Aquaporin-4 (AQP4) Polarized Perivascular Clearance Invariants
Astrocytic endfoot AQP4 water transport polarization, NREM slow-wave sleep interstitial volume expansion (alpha),
arterial low-frequency vasomotion, convective bulk flow (Darcy's law), and parenchymal amyloid/fibrin clearance.
-/

/-- Effective AQP4 polarization percentage (15 - 100)
    eff_pol = min(100, max(15, pol_pct + (agrin_pct * 55 / 100)))
-/
def compute_effective_aqp4_polarization (pol_pct : Nat) (agrin_pct : Nat) : Nat :=
  let boost := (agrin_pct * 55) / 100
  let raw := pol_pct + boost
  if raw > 100 then 100 else (if raw < 15 then 15 else raw)

/-- Extracellular volume fraction alpha percentage scaled x10 (140 = 14.0%, 240 = 24.0%) -/
def compute_interstitial_volume_fraction (delta_power_x100 : Nat) (shed_dose : Nat) : Nat :=
  let raw := 140 + (delta_power_x100 * 10) / 10
  let dose_boost := (shed_dose * 20) / 100
  let total := (raw * (100 + dose_boost)) / 100
  if total > 250 then 250 else (if total < 120 then 120 else total)

/-- Convective interstitial flux velocity in um/min scaled x100 (180 = 1.80 um/min)
    flux = (vasomotion_x10 * 10 / 35) * (eff_pol * 100 / 85) * (alpha_x10 * 100 / 220) * 210 / 10000
-/
def compute_convective_flux (vasomotion_x10 : Nat) (eff_pol : Nat) (alpha_x10 : Nat) : Nat :=
  let v_part := (vasomotion_x10 * 100) / 35
  let p_part := (eff_pol * 100) / 85
  let a_part := (alpha_x10 * 100) / 220
  let raw := (v_part * p_part * a_part * 210) / 1000000
  if raw < 10 then 10 else (if raw > 350 then 350 else raw)

/-- Parenchymal amyloid and fibrin clearance percentage per hour (0 - 100) -/
def compute_solute_clearance_rate (flux_x100 : Nat) : Nat :=
  let raw := (flux_x100 * 75) / 200
  if raw > 100 then 100 else (if raw < 5 then 5 else raw)

/-- Toxic parenchymal solute retention percentage (0 - 100) -/
def compute_parenchymal_retention (clearance_pct : Nat) : Nat :=
  if clearance_pct ≥ 100 then 0 else 100 - clearance_pct

theorem impaired_glymphatics_drives_severe_solute_retention :
  let eff_pol := compute_effective_aqp4_polarization 25 0
  let alpha := compute_interstitial_volume_fraction 18 0
  let flux := compute_convective_flux 9 eff_pol alpha
  let clearance := compute_solute_clearance_rate flux
  let retention := compute_parenchymal_retention clearance
  flux ≤ 50 ∧ clearance ≤ 25 ∧ retention ≥ 75 := by
  dsimp [compute_effective_aqp4_polarization, compute_interstitial_volume_fraction,
         compute_convective_flux, compute_solute_clearance_rate, compute_parenchymal_retention]
  decide

theorem shed_cm_infusion_fully_reconstitutes_glymphatic_clearance :
  let eff_pol := compute_effective_aqp4_polarization 35 100
  let alpha := compute_interstitial_volume_fraction 85 100
  let flux := compute_convective_flux 38 eff_pol alpha
  let clearance := compute_solute_clearance_rate flux
  let retention := compute_parenchymal_retention clearance
  flux ≥ 180 ∧ clearance ≥ 70 ∧ retention ≤ 30 := by
  dsimp [compute_effective_aqp4_polarization, compute_interstitial_volume_fraction,
         compute_convective_flux, compute_solute_clearance_rate, compute_parenchymal_retention]
  decide

end Weft.Formal.GlymphaticAqp4
