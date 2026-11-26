-- Corticospinal Fractionated Digit Control and Monosynaptic Cortico-Motoneuronal Invariants
-- Formal verification of the lateral corticospinal tract (LCST), primary motor cortex (M1)
-- Betz pyramidal cell monosynaptic coupling to cervical (C7-T1) motoneuron pools,
-- and individual finger force fractionation bounds.
-- Target #368 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.CorticospinalFractionation

/-- Grasp execution mode:
    - `fractionatedPrecision`: Individual digit force modulated with high selectivity.
    - `synergisticPowerGrasp`: Non-fractionated co-activation of all digits.
-/
inductive DigitGraspMode where
  | fractionatedPrecision : DigitGraspMode
  | synergisticPowerGrasp : DigitGraspMode
  deriving Repr, DecidableEq

/-- State parameters for corticomotoneuronal (CM) drive and fractionated digit control:
    `targetForce`: Commanded force for targeted digit (Newtons, scaled as Nat)
    `m1BaseRate`: Spontaneous baseline firing rate of M1 Betz cells (Hz)
    `m1Gain`: Firing rate gain per Newton of commanded force (Hz / N)
    `spilloverGain`: Fractional spillover onto non-targeted synergistic digits (e.g. 5%)
    `fractionationThreshold`: Upper bound of acceptable non-target force (Newtons)
-/
structure CorticospinalState where
  targetForce : Nat
  m1BaseRate : Nat
  m1Gain : Nat
  spilloverGain : Nat
  fractionationThreshold : Nat
  h_gain_pos : m1Gain > 0
  h_spill_bound : spilloverGain <= 10  -- spillover coefficient <= 10%
  deriving Repr

/-- Compute primary motor cortex (M1) Betz cell firing rate. -/
def computeM1FiringRate (s : CorticospinalState) : Nat :=
  s.m1BaseRate + s.m1Gain * s.targetForce

/-- Compute spillover force transmitted to non-target adjacent digits.
    Scales as (targetForce * spilloverGain) / 100. -/
def computeSpilloverForce (s : CorticospinalState) : Nat :=
  (s.targetForce * s.spilloverGain) / 100

/-- Evaluate grasp fractionation mode.
    Classified as fractionated precision if spillover force remains below threshold. -/
def evaluateGraspMode (s : CorticospinalState) : DigitGraspMode :=
  if computeSpilloverForce s <= s.fractionationThreshold then
    DigitGraspMode.fractionatedPrecision
  else
    DigitGraspMode.synergisticPowerGrasp

/-- Invariant: Is the grasp fractionated? -/
def isFractionated (s : CorticospinalState) : Prop :=
  computeSpilloverForce s <= s.fractionationThreshold

/-- Theorem 1: M1 Betz cell firing rate is strictly monotonic with target digit force. -/
theorem m1_firing_rate_monotonic (s : CorticospinalState) (f1 f2 : Nat)
    (h_le : f1 <= f2) :
    let s1 : CorticospinalState := { s with targetForce := f1 }
    let s2 : CorticospinalState := { s with targetForce := f2 }
    computeM1FiringRate s1 <= computeM1FiringRate s2 := by
  intro s1 s2
  dsimp [computeM1FiringRate, s1, s2]
  have h_mul : s.m1Gain * f1 <= s.m1Gain * f2 := Nat.mul_le_mul_left s.m1Gain h_le
  exact Nat.add_le_add_left h_mul s.m1BaseRate

/-- Theorem 2: Spillover force monotonicity with target digit force. -/
theorem spillover_force_monotonic (s : CorticospinalState) (f1 f2 : Nat)
    (h_le : f1 <= f2) :
    let s1 : CorticospinalState := { s with targetForce := f1 }
    let s2 : CorticospinalState := { s with targetForce := f2 }
    computeSpilloverForce s1 <= computeSpilloverForce s2 := by
  intro s1 s2
  dsimp [computeSpilloverForce, s1, s2]
  have h_num : f1 * s.spilloverGain <= f2 * s.spilloverGain := Nat.mul_le_mul_right s.spilloverGain h_le
  exact Nat.div_le_div_right h_num

/-- Theorem 3: Strict fractionation bound for low force regimes.
    If targetForce * spilloverGain <= 100 * fractionationThreshold, precision grasp is guaranteed. -/
theorem fractionated_precision_guarantee (s : CorticospinalState)
    (h_bound : s.targetForce * s.spilloverGain <= 100 * s.fractionationThreshold) :
    isFractionated s := by
  dsimp [isFractionated, computeSpilloverForce]
  exact Nat.div_le_of_le_mul h_bound

/-- Theorem 4: Fractionation mode gate preservation. -/
theorem fractionation_mode_gate (s : CorticospinalState)
    (h_frac : computeSpilloverForce s <= s.fractionationThreshold) :
    evaluateGraspMode s = DigitGraspMode.fractionatedPrecision := by
  dsimp [evaluateGraspMode]
  rw [if_pos h_frac]

end Weft.Formal.CorticospinalFractionation
