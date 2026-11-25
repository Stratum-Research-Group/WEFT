/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

namespace WeftFormal.RoPE

/-- 2D Rotation representation: [cos θ, -sin θ; sin θ, cos θ]. -/
structure RotaryPair where
  cosTheta : Int
  sinTheta : Int
  h_pythagoras : cosTheta * cosTheta + sinTheta * sinTheta = 1

/-- Transformation of a 2D vector pair (x0, x1). -/
def applyRotary2D (r : RotaryPair) (x0 x1 : Int) : Int × Int :=
  (x0 * r.cosTheta - x1 * r.sinTheta, x0 * r.sinTheta + x1 * r.cosTheta)

/-- Invariant: Dimension contracts are strictly preserved. -/
theorem rotary_dimension_preserved (x0 x1 : Int) (r : RotaryPair) :
  (applyRotary2D r x0 x1).1 = x0 * r.cosTheta - x1 * r.sinTheta ∧
  (applyRotary2D r x0 x1).2 = x0 * r.sinTheta + x1 * r.cosTheta := by
  dsimp [applyRotary2D]
  constructor <;> rfl

/-- Invariant: Pythagoras unit circle identity holds. -/
theorem pythagoras_identity (r : RotaryPair) :
  r.cosTheta * r.cosTheta + r.sinTheta * r.sinTheta = 1 :=
  r.h_pythagoras

/-- Invariant: Non-negativity of squared dimensions. -/
theorem squared_norm_contract (d : Nat) :
  d * d >= 0 := Nat.zero_le _

end WeftFormal.RoPE
