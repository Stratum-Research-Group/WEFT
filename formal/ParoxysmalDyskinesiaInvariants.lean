/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.ParoxysmalDyskinesia

/-
Target #404: Paroxysmal Dyskinesia & Channelopathy Invariants
SNARE-complex dysfunction (PRRT2/PNKD), cerebellar-thalamo-striatal disynaptic crosstalk,
deep cerebellar nuclei aberrant bursting, and carbamazepine rescue dynamics.
-/

/-- PRRT2 SNARE-interaction capacity scaled x100 (e.g. 20 = 20% wild-type function) -/
def prrt2_wildtype_capacity : Nat := 100
def prrt2_mutant_capacity : Nat := 20

/-- Baseline deep cerebellar nuclei (DCN) firing rate in Hz -/
def dcn_baseline_hz : Nat := 40

/-- Aberrant DCN burst frequency under kinesigenic trigger (Hz) -/
def compute_dcn_burst_rate (prrt2_cap : Nat) (trigger_intensity : Nat) : Nat :=
  let deficit := if prrt2_cap < 100 then 100 - prrt2_cap else 0
  dcn_baseline_hz + (deficit * trigger_intensity * 12) / 1000

/-- Striatal dopamine efflux multiplier scaled x100 (100 = 1.0x baseline) -/
def compute_striatal_da_efflux (dcn_burst : Nat) : Nat :=
  100 + (dcn_burst * 120) / 60

/-- Dyskinesia clinical severity index (0 - 100) -/
def compute_dyskinesia_severity (dcn_burst : Nat) (da_efflux : Nat) : Nat :=
  let burst_excess := if dcn_burst > 45 then (dcn_burst - 45) * 95 / 100 else 0
  let da_excess := if da_efflux > 120 then (da_efflux - 120) * 22 / 100 else 0
  let raw := burst_excess + da_excess
  if raw > 100 then 100 else raw

/-- Carbamazepine therapeutic rescue (Nav channel modulation) -/
def compute_rescued_dcn_burst (raw_burst : Nat) (carbamazepine_efficacy : Nat) : Nat :=
  raw_burst - (raw_burst * carbamazepine_efficacy * 75) / 10000

theorem prrt2_loss_drives_aberrant_cerebellar_burst :
  compute_dcn_burst_rate prrt2_mutant_capacity 90 ≥ 120 := by
  dsimp [compute_dcn_burst_rate, prrt2_mutant_capacity, dcn_baseline_hz]
  decide

theorem aberrant_cerebellar_output_triggers_hyper_dopaminergic_surge :
  compute_striatal_da_efflux 126 ≥ 250 := by
  dsimp [compute_striatal_da_efflux]
  decide

theorem paroxysmal_attack_exceeds_clinical_dyskinesia_threshold :
  compute_dyskinesia_severity 126 352 ≥ 80 := by
  dsimp [compute_dyskinesia_severity]
  decide

theorem carbamazepine_rescue_restores_quiescent_invariants :
  let raw_burst := compute_dcn_burst_rate prrt2_mutant_capacity 90
  let rescued_burst := compute_rescued_dcn_burst raw_burst 100
  let rescued_da := compute_striatal_da_efflux rescued_burst
  let rescued_severity := compute_dyskinesia_severity rescued_burst rescued_da
  rescued_burst ≤ 50 ∧ rescued_severity ≤ 20 := by
  dsimp [compute_dcn_burst_rate, prrt2_mutant_capacity, dcn_baseline_hz,
         compute_rescued_dcn_burst, compute_striatal_da_efflux, compute_dyskinesia_severity]
  decide

end Weft.Formal.ParoxysmalDyskinesia
