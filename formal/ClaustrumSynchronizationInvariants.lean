-- Claustrum Cortical Synchronization & Salience Switching Invariants
-- Formal verification of claustral widespread feedforward inhibition and cross-area phase resets:
-- 1. Crick & Koch (2005) "Conductor of Consciousness" & Atlan et al. (2018) Salience Switching.
-- 2. Monotonic escalation of claustral burst firing by salience perturbation amplitude.
-- 3. Widespread feedforward GABAergic inhibition of neocortical pyramidal populations via PV+ interneurons.
-- 4. Bounded cortical synchrony index [0, 100].
-- 5. Cognitive interrupt threshold triggering task switching.
-- Target #393 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.ClaustrumSynchronization

/-- State parameters representing claustral projections and recipient cortical networks.
    All firing rates, gains, and synchrony metrics are scaled natural units. -/
structure ClaustrumState where
  /-- Salience perturbation amplitude (scaled Nat, 0 - 100) -/
  salienceInput : Nat
  /-- Ongoing task engagement drive (scaled Nat, 0 - 100) -/
  taskDrive : Nat
  /-- Baseline claustral tonic firing rate (scaled Nat, e.g. 20 representing 2.0 Hz) -/
  claustralBaseline : Nat
  /-- Interneuron coupling efficiency (scaled Nat, e.g. 50 representing 0.50) -/
  interneuronGain : Nat
  deriving Repr

/-- Effective Claustral Projection Firing Rate:
    Rate_cl = claustralBaseline + (85 * salienceInput) / 100 -/
def claustralFiringRate (s : ClaustrumState) : Nat :=
  s.claustralBaseline + (85 * s.salienceInput) / 100

/-- Widespread Cortical Feedforward Inhibition (via PV+ interneurons):
    Inh = 10 + (interneuronGain * claustralFiringRate) / 100 (scaled integer, base 10 = 1.0x) -/
def feedforwardInhibition (s : ClaustrumState) : Nat :=
  10 + (s.interneuronGain * claustralFiringRate s) / 100

/-- Cortical Network Synchrony Index (scaled Nat, 0 - 100):
    Sync = min(100, 10 + (65 * claustralFiringRate) / 100) -/
def networkSynchronyIndex (s : ClaustrumState) : Nat :=
  let raw := 10 + (65 * claustralFiringRate s) / 100
  if raw <= 100 then raw else 100

/-- Net Salience Interrupt Drive:
    Drive_switch = if salienceInput > taskDrive then salienceInput - taskDrive else 0 -/
def salienceInterruptDrive (s : ClaustrumState) : Nat :=
  if s.salienceInput > s.taskDrive then s.salienceInput - s.taskDrive else 0

/-- Theorem 1: Claustral firing rate is bounded from below by baseline. -/
theorem claustral_firing_bounded_by_baseline (s : ClaustrumState) :
    claustralFiringRate s >= s.claustralBaseline := by
  dsimp [claustralFiringRate]
  omega

/-- Theorem 2: Elevating salience perturbation monotonically increases claustral firing rate. -/
theorem salience_monotone_claustral_rate (s : ClaustrumState) (sal1 sal2 : Nat)
    (h_sal : sal1 <= sal2) :
    claustralFiringRate { s with salienceInput := sal1 } <=
    claustralFiringRate { s with salienceInput := sal2 } := by
  dsimp [claustralFiringRate]
  apply Nat.add_le_add_left
  apply Nat.div_le_div_right
  omega

/-- Theorem 3: Feedforward inhibition is strictly positive (>= 10, i.e. 1.0x baseline). -/
theorem feedforward_inhibition_strictly_positive (s : ClaustrumState) :
    feedforwardInhibition s >= 10 := by
  dsimp [feedforwardInhibition]
  omega

/-- Theorem 4: Network synchrony index is bounded within [10, 100]. -/
theorem synchrony_index_bounded (s : ClaustrumState) :
    networkSynchronyIndex s >= 10 /\ networkSynchronyIndex s <= 100 := by
  dsimp [networkSynchronyIndex]
  split <;> omega

/-- Theorem 5: When task engagement dominates (taskDrive >= salienceInput), salience interrupt drive collapses to 0. -/
theorem strong_task_engagement_suppresses_interrupt (s : ClaustrumState)
    (h_task : s.taskDrive >= s.salienceInput) :
    salienceInterruptDrive s = 0 := by
  dsimp [salienceInterruptDrive]
  split
  · omega
  · rfl

end Weft.Formal.ClaustrumSynchronization
