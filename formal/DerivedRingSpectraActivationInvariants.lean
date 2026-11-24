-- DerivedRingSpectraActivationInvariants.lean
-- Milestone Paper 258: Asymmetric Derived Ring Spectra & Topological André-Quillen Cohomology
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure ActivationCell where
  cellId : Nat
  xLow : Nat
  xHigh : Nat
  slope : Nat
  bias : Nat
  h_bound : xLow <= xHigh
  deriving Repr

def evalLinearActivation (c : ActivationCell) (x : Nat) : Nat :=
  c.slope * x + c.bias

theorem linear_activation_nonneg (c : ActivationCell) (x : Nat) :
    evalLinearActivation c x >= 0 := by
  exact Nat.zero_le _

theorem linear_activation_mono (c : ActivationCell) (x y : Nat) (h_le : x <= y) :
    evalLinearActivation c x <= evalLinearActivation c y := by
  dsimp [evalLinearActivation]
  have h_mul : c.slope * x <= c.slope * y := Nat.mul_le_mul_left c.slope h_le
  omega

theorem cell_interval_nonneg (c : ActivationCell) :
    c.xHigh - c.xLow >= 0 := by
  exact Nat.zero_le _

theorem linear_activation_bias_bound (c : ActivationCell) (x : Nat) :
    evalLinearActivation c x >= c.bias := by
  dsimp [evalLinearActivation]
  omega
