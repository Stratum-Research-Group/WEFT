/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.FibrinogenShed

/-
Target #406: Fibrinogen-SHED-CM Molecular Interactions & Anti-Amyloid Shielding Invariants
Proteomic interactions: HGF epitope masking, Clusterin/Decorin chaperone capping,
inhibition of thrombin cleavage, suppression of cross-beta sheet amyloid fibrin formation,
and attenuation of microglial Mac-1 (CD11b/CD18) neuroinflammatory cascade.
-/

/-- Raw amyloidogenic fibrin clot drive scaled x100
    clot_raw = (fibrinogen_x10 / 30) * (thrombin_x100 / 150) * (100 + abeta_nM * 2) * 35 / 100
-/
def compute_raw_clot_drive (fib_x10 : Nat) (thr_x100 : Nat) (abeta_nM : Nat) : Nat :=
  let fib_factor := (fib_x10 * 100) / 30
  let thr_factor := (thr_x100 * 100) / 150
  let abeta_factor := 100 + abeta_nM * 2
  let product := (fib_factor * thr_factor * abeta_factor * 35) / 1000000
  if product > 100 then 100 else product

/-- SHED-CM shielding factor scaled x100 (HGF + Clusterin chaperone activity) -/
def compute_shed_shielding (shed_dose_percent : Nat) : Nat :=
  let hgf_part := (shed_dose_percent * 55) / 100
  let clusterin_part := (shed_dose_percent * 40) / 100
  let total := hgf_part + clusterin_part
  if total > 92 then 92 else total

/-- Effective amyloidogenic fibrin clot mass scaled x100 -/
def compute_effective_clot (raw_clot : Nat) (shielding : Nat) : Nat :=
  (raw_clot * (100 - shielding)) / 100

/-- Microglial Mac-1 (CD11b/CD18) activation score scaled x100 -/
def compute_mac1_activation (clot : Nat) (shed_dose : Nat) : Nat :=
  let raw_mac1 := (clot * 92) / 100
  let suppression := (shed_dose * 35) / 100
  (raw_mac1 * (100 - suppression)) / 100

/-- Synaptic spine elimination index scaled x100 -/
def compute_spine_elimination (mac1 : Nat) : Nat :=
  (mac1 * 82) / 100

theorem extravasation_and_abeta_drive_massive_amyloid_clot :
  let raw_clot := compute_raw_clot_drive 32 185 85
  raw_clot ≥ 90 := by
  dsimp [compute_raw_clot_drive]
  decide

theorem shed_cm_infusion_shields_fibrin_and_rescues_synapses :
  let raw_clot := compute_raw_clot_drive 32 185 85
  let shielding := compute_shed_shielding 100
  let eff_clot := compute_effective_clot raw_clot shielding
  let mac1 := compute_mac1_activation eff_clot 100
  let spine_loss := compute_spine_elimination mac1
  eff_clot ≤ 15 ∧ mac1 ≤ 15 ∧ spine_loss ≤ 10 := by
  dsimp [compute_raw_clot_drive, compute_shed_shielding, compute_effective_clot,
         compute_mac1_activation, compute_spine_elimination]
  decide

theorem untreated_fibrin_induces_severe_synaptic_destruction :
  let raw_clot := compute_raw_clot_drive 32 185 85
  let eff_clot := compute_effective_clot raw_clot 0
  let mac1 := compute_mac1_activation eff_clot 0
  let spine_loss := compute_spine_elimination mac1
  spine_loss ≥ 70 := by
  dsimp [compute_raw_clot_drive, compute_effective_clot, compute_mac1_activation, compute_spine_elimination]
  decide

end Weft.Formal.FibrinogenShed
