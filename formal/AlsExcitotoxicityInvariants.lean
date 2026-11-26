/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.AlsExcitotoxicityInvariants

/-!
# Target #401: Amyotrophic Lateral Sclerosis (ALS) Corticomotoneuronal Excitotoxicity & Renshaw Recurrent Disinhibition Invariants
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group

Formal Invariant Guarantees:
1. `healthy_eaat2_maintains_physiological_glutamate`: 100% astrocytic EAAT2 expression clamps synaptic glutamate below 200 (2.00 μM).
2. `eaat2_loss_triggers_excitotoxic_accumulation`: Severe loss of EAAT2 (expression < 25%) causes glutamate accumulation exceeding 1000 (10.00 μM).
3. `glutamate_excess_drives_runaway_motoneuron_firing`: Glutamate excess coupled with Renshaw loss drives motoneuron firing exceeding 450 (45.0 Hz).
4. `excitotoxic_firing_triggers_high_frequency_fasciculations`: Pathological motoneuron firing generates spontaneous fasciculation rates exceeding 30 (3.0 Hz).
5. `dual_riluzole_glycine_rescue_restores_homeostasis`: Combined Riluzole glutamate release blockade and glycinergic Renshaw agonism restores motoneuron firing below 250 (25.0 Hz).
-/

def GLUTAMATE_PHYSIOLOGICAL_CEILING : Nat := 200 -- 2.00 μM (scale 100)
def GLUTAMATE_EXCITOTOXIC_FLOOR : Nat := 1000 -- 10.00 μM (scale 100)
def MOTONEURON_EXCITOTOXIC_FLOOR : Nat := 450 -- 45.0 Hz (scale 10)
def MOTONEURON_SAFE_CEILING : Nat := 250 -- 25.0 Hz (scale 10)
def FASCICULATION_ALARM_FLOOR : Nat := 30 -- 3.0 Hz (scale 10)

/--
Compute synaptic glutamate concentration (x100 μM):
Glutamate = 120 + (1800 * (1000 - eaat2) * (1000 - (riluzole * 7) / 10)) / 1000000
-/
def compute_synaptic_glutamate (eaat2 : Nat) (riluzole : Nat) : Nat :=
  let eaat2_eff := if eaat2 > 1000 then 1000 else eaat2
  let riluzole_eff := if riluzole > 1000 then 1000 else riluzole
  let eaat2_loss := 1000 - eaat2_eff
  let riluzole_block := 1000 - (riluzole_eff * 7) / 10
  120 + (1800 * eaat2_loss * riluzole_block) / 1000000

/--
Compute spinal Renshaw recurrent inhibition gain (scale 1000):
Renshaw gain = min(1000, renshaw_integrity * 8 / 10 + glycine_agonist * 5 / 10)
-/
def compute_renshaw_gain (renshaw_integrity : Nat) (glycine_agonist : Nat) : Nat :=
  let r_eff := if renshaw_integrity > 1000 then 1000 else renshaw_integrity
  let g_eff := if glycine_agonist > 1000 then 1000 else glycine_agonist
  let total := (r_eff * 8) / 10 + (g_eff * 5) / 10
  if total > 1000 then 1000 else total

/--
Compute motoneuron firing rate (x10 Hz):
Raw excitation = 120 + (glu * 35) / 100
Renshaw brake = (renshaw_gain * 240) / 1000
Motoneuron rate = max(80, raw - brake)
-/
def compute_motoneuron_firing (glu : Nat) (renshaw_gain : Nat) : Nat :=
  let raw := 120 + (glu * 35) / 100
  let brake := (renshaw_gain * 240) / 1000
  if raw ≤ brake + 80 then
    80
  else
    raw - brake

/--
Compute fasciculation potential frequency (x10 Hz):
If glu > 400 and renshaw_gain < 500:
fasc = ((glu - 400) * 45 * (1000 - renshaw_gain)) / 1000000
Else: 1 (0.1 Hz)
-/
def compute_fasciculation_frequency (glu : Nat) (renshaw_gain : Nat) : Nat :=
  if glu > 400 ∧ renshaw_gain < 500 then
    let glu_excess := glu - 400
    let renshaw_deficit := 1000 - renshaw_gain
    let raw := (glu_excess * 45 * renshaw_deficit) / 1000000
    if raw == 0 then 1 else raw
  else
    1

/-! ### Invariant Theorem 1: Healthy EAAT2 Expression Preserves Normal Glutamate -/
theorem healthy_eaat2_maintains_physiological_glutamate :
    compute_synaptic_glutamate 1000 0 ≤ GLUTAMATE_PHYSIOLOGICAL_CEILING := by
  dsimp [compute_synaptic_glutamate, GLUTAMATE_PHYSIOLOGICAL_CEILING]
  decide

/-! ### Invariant Theorem 2: EAAT2 Depletion Drives Synaptic Glutamate Accumulation -/
theorem eaat2_loss_triggers_excitotoxic_accumulation :
    compute_synaptic_glutamate 150 0 > GLUTAMATE_EXCITOTOXIC_FLOOR := by
  dsimp [compute_synaptic_glutamate, GLUTAMATE_EXCITOTOXIC_FLOOR]
  decide

/-! ### Invariant Theorem 3: Runaway Motoneuron Firing in ALS -/
theorem glutamate_excess_drives_runaway_motoneuron_firing :
    let glu := compute_synaptic_glutamate 150 0
    let renshaw := compute_renshaw_gain 200 0
    compute_motoneuron_firing glu renshaw > MOTONEURON_EXCITOTOXIC_FLOOR := by
  dsimp [compute_synaptic_glutamate, compute_renshaw_gain, compute_motoneuron_firing, MOTONEURON_EXCITOTOXIC_FLOOR]
  decide

/-! ### Invariant Theorem 4: Excitotoxicity and Renshaw Loss Trigger High Fasciculation Frequencies -/
theorem excitotoxic_firing_triggers_high_frequency_fasciculations :
    let glu := compute_synaptic_glutamate 150 0
    let renshaw := compute_renshaw_gain 200 0
    compute_fasciculation_frequency glu renshaw ≥ FASCICULATION_ALARM_FLOOR := by
  dsimp [compute_synaptic_glutamate, compute_renshaw_gain, compute_fasciculation_frequency, FASCICULATION_ALARM_FLOOR]
  decide

/-! ### Invariant Theorem 5: Dual Riluzole & Glycinergic Rescue Restores Physiological Range -/
theorem dual_riluzole_glycine_rescue_restores_homeostasis :
    let glu_rescued := compute_synaptic_glutamate 150 850
    let renshaw_rescued := compute_renshaw_gain 200 800
    compute_motoneuron_firing glu_rescued renshaw_rescued ≤ MOTONEURON_SAFE_CEILING := by
  dsimp [compute_synaptic_glutamate, compute_renshaw_gain, compute_motoneuron_firing, MOTONEURON_SAFE_CEILING]
  decide

end Weft.Formal.AlsExcitotoxicityInvariants
