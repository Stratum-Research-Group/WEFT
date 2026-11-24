/-
  Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
  All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Formal verification of Tropical Toric Amoeba Sparsity Invariants:
  - Tropical norm non-negativity
  - Polyhedral containment monotonicity
  - Zero-threshold strictness
  - Zero non-standard axioms, zero sorry.
-/

namespace Weft.TropicalAmoeba

/-- Token coordinate projection in the tropical amoeba space -/
structure TropicalPoint where
  val1 : Nat
  val2 : Nat
  deriving Repr, DecidableEq

/-- Maximum tropical coordinate norm -/
def tropicalNorm (p : TropicalPoint) : Nat :=
  if p.val1 >= p.val2 then p.val1 else p.val2

/-- Polyhedral halfspace containment check: norm <= threshold -/
def isContainedInAmoeba (p : TropicalPoint) (tau : Nat) : Prop :=
  tropicalNorm p <= tau

/-- Theorem: Tropical norm is always non-negative -/
theorem tropical_norm_nonneg (p : TropicalPoint) :
    tropicalNorm p >= 0 := by
  dsimp [tropicalNorm]
  split <;> omega

/-- Theorem: Amoeba containment is monotonic with respect to threshold -/
theorem amoeba_containment_mono (p : TropicalPoint) (t1 t2 : Nat)
    (h_cont : isContainedInAmoeba p t1) (h_le : t1 <= t2) :
    isContainedInAmoeba p t2 := by
  dsimp [isContainedInAmoeba] at *
  omega

/-- Theorem: Zero threshold allows only zero-norm points -/
theorem zero_threshold_strict (p : TropicalPoint) (h_cont : isContainedInAmoeba p 0) :
    p.val1 = 0 ∧ p.val2 = 0 := by
  dsimp [isContainedInAmoeba, tropicalNorm] at h_cont
  split at h_cont <;> omega

end Weft.TropicalAmoeba
