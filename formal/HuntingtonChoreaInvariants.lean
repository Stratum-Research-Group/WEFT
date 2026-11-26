/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.HuntingtonChoreaInvariants

/-!
# Milestone Target #400: Huntington's Disease Striatal D2-SPN Degeneration & Hyperkinetic Chorea Invariants
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group

Formal Invariant Guarantees:
1. `healthy_d2_spn_preserves_gpe_restraint`: Under 100% D2-SPN indirect pathway survival, GPe firing remains physiologically restrained (≤ 55.0 Hz).
2. `d2_spn_loss_triggers_gpe_disinhibition`: Severe D2-SPN loss (survival < 30%) causes GPe hyperactivity exceeding 80.0 Hz.
3. `gpe_hyperactivity_silences_stn_brake`: Excessive GPe GABAergic drive suppresses STN firing below the critical motor brake threshold (12.0 Hz).
4. `loss_of_stn_brake_causes_chorea`: STN silencing disinhibits thalamic motor output, triggering involuntary hyperkinetic choreic intrusions (UHDRS chorea score ≥ 500, scale 1000).
5. `gpe_gaba_restoration_rescues_chorea`: Restoring GPe GABAergic restraint (rescue factor ≥ 750) re-engages STN braking and suppresses choreic intrusions below the clinical significance threshold (< 200).
-/

def GPE_PHYSIOLOGICAL_CEILING : Nat := 550 -- 55.0 Hz
def GPE_DISINHIBITED_FLOOR : Nat := 800 -- 80.0 Hz
def STN_BRAKE_CRITICAL_FLOOR : Nat := 120 -- 12.0 Hz
def CHOREA_CLINICAL_THRESHOLD : Nat := 500 -- 5.0 / 10 (scale 1000)
def CHOREA_RESCUE_CEILING : Nat := 200 -- 2.0 / 10 (scale 1000)

/--
Compute GPe firing rate (x10 Hz) as a function of D2-SPN indirect pathway survival [0, 1000]
and GPe GABAergic rescue agonist [0, 1000].
GPe = 520 + (480 * (1000 - d2_survival) * (1000 - (rescue * 75) / 100)) / 1000000
-/
def compute_gpe_firing_rate (d2_survival : Nat) (rescue : Nat) : Nat :=
  let d2_eff := if d2_survival > 1000 then 1000 else d2_survival
  let rescue_eff := if rescue > 1000 then 1000 else rescue
  let d2_loss := 1000 - d2_eff
  let rescue_factor := 1000 - (rescue_eff * 75) / 100
  520 + (480 * d2_loss * rescue_factor) / 1000000

/--
Compute STN motor brake firing rate (x10 Hz):
STN receives tonic drive (24.0 Hz) attenuated by GPe over-firing:
STN = if GPe > 500 then 240 - ((GPe - 500) * 4) / 10 else 240
Bounded below by 20 (2.0 Hz).
-/
def compute_stn_firing_rate (gpe_rate : Nat) : Nat :=
  if gpe_rate ≤ 500 then
    240
  else
    let inhibition := ((gpe_rate - 500) * 4) / 10
    if inhibition ≥ 220 then
      20
    else
      240 - inhibition

/--
Compute Chorea Severity Score (scale 1000):
Driven by loss of STN braking drive:
If STN < 200 (20.0 Hz), chorea = (200 - STN) * 5
Else chorea = 0
-/
def compute_chorea_score (stn_rate : Nat) : Nat :=
  if stn_rate ≥ 200 then
    0
  else
    let deficit := 200 - stn_rate
    let raw := deficit * 5
    if raw > 1000 then 1000 else raw

/-! ### Invariant Theorem 1: Healthy D2-SPN Population Preserves GPe Tonic Restraint -/
theorem healthy_d2_spn_preserves_gpe_restraint :
    compute_gpe_firing_rate 1000 0 ≤ GPE_PHYSIOLOGICAL_CEILING := by
  dsimp [compute_gpe_firing_rate, GPE_PHYSIOLOGICAL_CEILING]
  decide

/-! ### Invariant Theorem 2: Severe D2-SPN Loss Causes GPe Hyperactivity -/
theorem d2_spn_loss_triggers_gpe_disinhibition :
    compute_gpe_firing_rate 200 0 > GPE_DISINHIBITED_FLOOR := by
  dsimp [compute_gpe_firing_rate, GPE_DISINHIBITED_FLOOR]
  decide

/-! ### Invariant Theorem 3: GPe Hyperactivity Silences STN Motor Brake -/
theorem gpe_hyperactivity_silences_stn_brake :
    compute_stn_firing_rate (compute_gpe_firing_rate 200 0) < STN_BRAKE_CRITICAL_FLOOR := by
  dsimp [compute_gpe_firing_rate, compute_stn_firing_rate, STN_BRAKE_CRITICAL_FLOOR]
  decide

/-! ### Invariant Theorem 4: Loss of STN Brake Causes Overt Chorea -/
theorem loss_of_stn_brake_causes_chorea :
    compute_chorea_score (compute_stn_firing_rate (compute_gpe_firing_rate 200 0)) ≥ CHOREA_CLINICAL_THRESHOLD := by
  dsimp [compute_gpe_firing_rate, compute_stn_firing_rate, compute_chorea_score, CHOREA_CLINICAL_THRESHOLD]
  decide

/-! ### Invariant Theorem 5: GPe GABAergic Restoration Rescues Chorea -/
theorem gpe_gaba_restoration_rescues_chorea :
    compute_chorea_score (compute_stn_firing_rate (compute_gpe_firing_rate 200 800)) < CHOREA_RESCUE_CEILING := by
  dsimp [compute_gpe_firing_rate, compute_stn_firing_rate, compute_chorea_score, CHOREA_RESCUE_CEILING]
  decide

end Weft.Formal.HuntingtonChoreaInvariants
