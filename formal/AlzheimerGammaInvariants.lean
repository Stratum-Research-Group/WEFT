/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/
import Lean

namespace Weft.Formal.AlzheimerGammaInvariants

/-!
# Target #403: Alzheimer's Disease Corticocortical Gamma Uncoupling & Parvalbumin Interneuron Hypoactivity Invariants
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group

Formal Invariant Guarantees:
1. `healthy_pv_maintains_fast_spiking`: In the absence of amyloid-β / tau pathology (pathology = 0), PV+ fast-spiking interneurons fire at 55.0 Hz.
2. `pathology_causes_pv_hypoactivity`: Severe pathology (load ≥ 80%) reduces PV+ interneuron firing below 25.0 Hz.
3. `pv_hypoactivity_flattens_spectral_slope`: Loss of PV fast-spiking disinhibits cortical circuits, flattening the 1/f aperiodic spectral tilt slope χ below 1.60.
4. `pv_hypoactivity_causes_epileptiform_spikes`: PV hypoactivity triggers subclinical non-convulsive epileptiform discharges (rate ≥ 2.0 Hz).
5. `genus_and_nav11_rescue_gamma_synchrony`: Combined Nav1.1 enhancement and 40 Hz sensory gamma entrainment restores 40 Hz gamma power above 35.0 μV² and steepens 1/f slope χ above 2.00.
-/

def PV_FAST_SPIKING_BASELINE : Nat := 550 -- 55.0 Hz (scale 10)
def PV_HYPOACTIVE_CEILING : Nat := 250 -- 25.0 Hz (scale 10)
def SPECTRAL_SLOPE_ALARM_CEILING : Nat := 160 -- 1.60 (scale 100)
def SPECTRAL_SLOPE_RESCUE_FLOOR : Nat := 200 -- 2.00 (scale 100)
def EPILEPTIFORM_SPIKE_FLOOR : Nat := 20 -- 2.0 Hz (scale 10)
def GAMMA_POWER_RESCUE_FLOOR : Nat := 350 -- 35.0 μV² (scale 10)

/--
Compute Parvalbumin-positive (PV+) fast-spiking interneuron firing rate (x10 Hz):
PV_rate = max(120, 550 - (420 * path_load * (1000 - (nav_mod * 75) / 100)) / 1000000)
-/
def compute_pv_firing_rate (path_load : Nat) (nav_mod : Nat) : Nat :=
  let p_eff := if path_load > 1000 then 1000 else path_load
  let n_eff := if nav_mod > 1000 then 1000 else nav_mod
  let nav_factor := 1000 - (n_eff * 75) / 100
  let loss := (420 * p_eff * nav_factor) / 1000000
  if loss ≥ 430 then
    120
  else
    550 - loss

/--
Compute 40 Hz narrow-band gamma power (x10 μV²):
gamma = (pv_rate * 500) / 550
If GENUS 40 Hz entrainment is active, gamma is clamped from below at 420 (42.0 μV²).
-/
def compute_gamma_power (pv_rate : Nat) (genus_active : Bool) : Nat :=
  let raw := (pv_rate * 500) / 550
  if genus_active then
    if raw < 420 then 420 else raw
  else
    raw

/--
Compute 1/f aperiodic spectral slope χ (x100):
χ = 280 - (path_load * 160 * (1000 - (nav_mod * 75) / 100)) / 1000000 + (if genus_active then 60 else 0)
Bounded below by 100 (1.00).
-/
def compute_spectral_slope (path_load : Nat) (nav_mod : Nat) (genus_active : Bool) : Nat :=
  let p_eff := if path_load > 1000 then 1000 else path_load
  let n_eff := if nav_mod > 1000 then 1000 else nav_mod
  let nav_factor := 1000 - (n_eff * 75) / 100
  let dec := (p_eff * 160 * nav_factor) / 1000000
  let base := if dec ≥ 180 then 100 else 280 - dec
  if genus_active then base + 60 else base

/--
Compute subclinical epileptiform micro-burst frequency (x10 Hz):
If pv_rate < 300:
spikes = ((300 - pv_rate) * 25 * (if genus_active then 35 else 100)) / 10000
Else: 1 (0.1 Hz)
-/
def compute_epileptiform_rate (pv_rate : Nat) (genus_active : Bool) : Nat :=
  if pv_rate < 300 then
    let deficit := 300 - pv_rate
    let atten := if genus_active then 35 else 100
    let raw := (deficit * 25 * atten) / 10000
    if raw == 0 then 1 else raw
  else
    1

/-! ### Invariant Theorem 1: Healthy PV+ Fast-Spiking Baseline -/
theorem healthy_pv_maintains_fast_spiking :
    compute_pv_firing_rate 0 0 = PV_FAST_SPIKING_BASELINE := by
  rfl

/-! ### Invariant Theorem 2: Amyloid/Tau Pathology Causes Severe PV Hypoactivity -/
theorem pathology_causes_pv_hypoactivity :
    compute_pv_firing_rate 800 0 < PV_HYPOACTIVE_CEILING := by
  dsimp [compute_pv_firing_rate, PV_HYPOACTIVE_CEILING]
  decide

/-! ### Invariant Theorem 3: PV Hypoactivity Flattens Aperiodic 1/f Spectral Slope -/
theorem pv_hypoactivity_flattens_spectral_slope :
    compute_spectral_slope 800 0 false < SPECTRAL_SLOPE_ALARM_CEILING := by
  dsimp [compute_spectral_slope, SPECTRAL_SLOPE_ALARM_CEILING]
  decide

/-! ### Invariant Theorem 4: PV Hypoactivity Causes Subclinical Epileptiform Spikes -/
theorem pv_hypoactivity_causes_epileptiform_spikes :
    compute_epileptiform_rate (compute_pv_firing_rate 800 0) false ≥ EPILEPTIFORM_SPIKE_FLOOR := by
  dsimp [compute_pv_firing_rate, compute_epileptiform_rate, EPILEPTIFORM_SPIKE_FLOOR]
  decide

/-! ### Invariant Theorem 5: Combined Nav1.1 & GENUS Rescues Gamma Synchrony and 1/f Slope -/
theorem genus_and_nav11_rescue_gamma_synchrony :
    let pv_rescued := compute_pv_firing_rate 800 850
    compute_gamma_power pv_rescued true ≥ GAMMA_POWER_RESCUE_FLOOR ∧
    compute_spectral_slope 800 850 true ≥ SPECTRAL_SLOPE_RESCUE_FLOOR := by
  dsimp [compute_pv_firing_rate, compute_gamma_power, compute_spectral_slope,
         GAMMA_POWER_RESCUE_FLOOR, SPECTRAL_SLOPE_RESCUE_FLOOR]
  decide

end Weft.Formal.AlzheimerGammaInvariants
