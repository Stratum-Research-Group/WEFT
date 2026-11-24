-- MotivicLFunctionsPrecisionInvariants.lean
-- Milestone Paper 254: Asymmetric Motivic L-Functions & Beilinson Regulators for Sub-Bit Precision Allocation
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure LayerMotive where
  layerIndex : Nat
  regulatorStiffness : Nat
  deriving Repr, DecidableEq

def allocatePrecision (baseBits : Nat) (stiffness : Nat) (scale : Nat) : Nat :=
  baseBits + stiffness * scale

theorem allocated_precision_ge_base (base stiff scale : Nat) :
    allocatePrecision base stiff scale >= base := by
  dsimp [allocatePrecision]
  omega

theorem allocated_precision_nonneg (base stiff scale : Nat) :
    allocatePrecision base stiff scale >= 0 := by
  exact Nat.zero_le _

theorem monotonicity_with_stiffness
    (base scale stiff1 stiff2 : Nat)
    (h_le : stiff1 <= stiff2) :
    allocatePrecision base stiff1 scale <= allocatePrecision base stiff2 scale := by
  dsimp [allocatePrecision]
  have h_mul : stiff1 * scale <= stiff2 * scale := Nat.mul_le_mul_right scale h_le
  omega

theorem precision_budget_conservation
    (base1 base2 s1 s2 scale : Nat) :
    allocatePrecision base1 s1 scale + allocatePrecision base2 s2 scale =
    (base1 + base2) + (s1 + s2) * scale := by
  dsimp [allocatePrecision]
  rw [Nat.add_mul s1 s2 scale]
  omega
