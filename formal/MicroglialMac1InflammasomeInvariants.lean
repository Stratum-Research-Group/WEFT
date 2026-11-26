/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.MicroglialMac1Inflammasome

/-
Target #409: Microglial Mac-1 (CD11b/CD18) Cryptic Epitope (gamma390-396) Neutralization
& NLRP3 Inflammasome Gating Invariants.
Dual-signal inflammasome activation:
Signal 1 (Priming): Fibrin Mac-1 ligation + TLR4 -> NF-kB nuclear translocation -> NLRP3/pro-IL-1b transcription.
Signal 2 (Assembly): P2X7 purinergic receptor ligation -> K+ efflux -> ASC speck oligomerization -> Caspase-1 p20 cleavage.
SHED-CM rescue: HGF steric epitope shielding (gamma377-395) + exosomal miR-124-3p silencing of ITGAM and MyD88.
-/

/-- Effective gamma390-396 epitope exposure scaled x100
    hgf_shielding = min(92, shed_dose * 95 / 100)
    eff_gamma = gamma_pct * (100 - hgf_shielding) / 100
-/
def compute_effective_gamma (gamma_pct : Nat) (shed_dose : Nat) : Nat :=
  let raw_shield := (shed_dose * 95) / 100
  let shield := if raw_shield > 92 then 92 else raw_shield
  (gamma_pct * (100 - shield)) / 100

/-- Microglial Mac-1 (CD11b/CD18) integrin active ligation percentage (0 - 100) -/
def compute_mac1_ligation (eff_gamma : Nat) (shed_dose : Nat) : Nat :=
  let silencing := (shed_dose * 65) / 100
  let raw := (eff_gamma * 105 * (100 - silencing)) / 10000
  if raw < 2 then 2 else (if raw > 100 then 100 else raw)

/-- NLRP3 ASC speck assembly percentage (0 - 100) -/
def compute_asc_speck_assembly (tlr4_primer : Nat) (mac1_pct : Nat) (p2x7_trigger : Nat) (shed_dose : Nat) : Nat :=
  let raw_priming := (tlr4_primer * 40 + mac1_pct * 60) / 100
  let priming_suppression := (shed_dose * 70) / 100
  let priming := (raw_priming * (100 - priming_suppression)) / 100
  let act_suppression := (shed_dose * 50) / 100
  let activation := (p2x7_trigger * (100 - act_suppression)) / 100
  let product := (priming * activation) / 75
  if product > 100 then 100 else product

/-- Active cleaved Caspase-1 p20 percentage (0 - 100) -/
def compute_caspase1_cleavage (speck_pct : Nat) : Nat :=
  (speck_pct * 94) / 100

/-- Mature IL-1beta secretion in pg/mL -/
def compute_il1beta_efflux (casp1_pct : Nat) : Nat :=
  let raw := 15 + (casp1_pct * 52) / 10
  if raw > 600 then 600 else raw

/-- Microglial M1/M2 phenotype ratio scaled x100 (100 = 1.0x) -/
def compute_m1_m2_ratio (mac1_pct : Nat) (shed_dose : Nat) : Nat :=
  let base := (mac1_pct * 100) / 15
  let suppression := (shed_dose * 88) / 100
  let eff := (base * (100 - suppression)) / 100
  if eff < 12 then 12 else eff

theorem unshielded_fibrin_triggers_massive_inflammasome_assembly :
  let eff_gamma := compute_effective_gamma 95 0
  let mac1 := compute_mac1_ligation eff_gamma 0
  let speck := compute_asc_speck_assembly 85 mac1 90 0
  let casp1 := compute_caspase1_cleavage speck
  let il1b := compute_il1beta_efflux casp1
  let ratio := compute_m1_m2_ratio mac1 0
  mac1 ≥ 85 ∧ speck ≥ 85 ∧ il1b ≥ 400 ∧ ratio ≥ 400 := by
  dsimp [compute_effective_gamma, compute_mac1_ligation, compute_asc_speck_assembly,
         compute_caspase1_cleavage, compute_il1beta_efflux, compute_m1_m2_ratio]
  decide

theorem shed_cm_infusion_quenches_inflammasome_and_drives_m2_polarization :
  let eff_gamma := compute_effective_gamma 95 100
  let mac1 := compute_mac1_ligation eff_gamma 100
  let speck := compute_asc_speck_assembly 85 mac1 90 100
  let casp1 := compute_caspase1_cleavage speck
  let il1b := compute_il1beta_efflux casp1
  let ratio := compute_m1_m2_ratio mac1 100
  mac1 ≤ 5 ∧ speck ≤ 15 ∧ il1b ≤ 90 ∧ ratio ≤ 30 := by
  dsimp [compute_effective_gamma, compute_mac1_ligation, compute_asc_speck_assembly,
         compute_caspase1_cleavage, compute_il1beta_efflux, compute_m1_m2_ratio]
  decide

end Weft.Formal.MicroglialMac1Inflammasome
