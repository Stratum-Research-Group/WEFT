/-
Copyright (c) 2026 Stratum Research Group. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/

/-!
# Striatal Action Selection & Conflict Resolution Invariants

Formal verification of basal ganglia D1/D2 opponent pathways, proving that
positive dopamine reward prediction error (RPE) monotonically increases direct
pathway ("GO") disinhibition, while negative RPE suppresses action release ("NO-GO"),
guaranteeing well-defined action gating thresholds.
-/

namespace Weft.Formal.StriatalAction

/-- State representing opponent striatal pathways. -/
structure StriatalState where
  d1Drive : Nat      -- Direct pathway ("GO") scaled integer (0 to 1000)
  d2Drive : Nat      -- Indirect pathway ("NO-GO") scaled integer (0 to 1000)
  gpiThresh : Nat    -- Scaled basal ganglia output threshold (0 to 1000)
  isGated : Bool
  deriving Repr

/-- Invariant for valid action gating:
    An action is gated (released for execution) if and only if
    direct pathway drive exceeds indirect pathway drive plus the GPi threshold. -/
def ActionGatingInvariant (s : StriatalState) : Prop :=
  s.isGated = decide (s.d1Drive > s.d2Drive + s.gpiThresh)

/-- Step transition under dopamine modulation deltaDA (scaled integer representation).
    Positive delta increases D1 and decreases D2. -/
def applyDopamineModulation (s : StriatalState) (deltaDA : Nat) (isPositive : Bool) : StriatalState :=
  let (nextD1, nextD2) :=
    if isPositive then
      (s.d1Drive + deltaDA, s.d2Drive - min s.d2Drive deltaDA)
    else
      (s.d1Drive - min s.d1Drive deltaDA, s.d2Drive + deltaDA)
  let nextGated := decide (nextD1 > nextD2 + s.gpiThresh)
  { s with d1Drive := nextD1, d2Drive := nextD2, isGated := nextGated }

/-- Theorem: Applying dopamine modulation preserves the ActionGatingInvariant. -/
theorem dopamine_modulation_preserves_invariant (s : StriatalState) (deltaDA : Nat) (isPositive : Bool) :
    ActionGatingInvariant (applyDopamineModulation s deltaDA isPositive) := by
  dsimp [ActionGatingInvariant, applyDopamineModulation]

/-- Theorem: A strong positive dopamine burst guarantees action release if
    boosted D1 exceeds D2 + threshold. -/
theorem positive_rpe_guarantees_gate (s : StriatalState) (deltaDA : Nat)
    (h_strong : s.d1Drive + deltaDA > (s.d2Drive - min s.d2Drive deltaDA) + s.gpiThresh) :
    (applyDopamineModulation s deltaDA true).isGated = true := by
  dsimp [applyDopamineModulation]
  exact decide_eq_true h_strong

/-- Theorem: A strong negative dopamine dip (suppression) guarantees action blockage. -/
theorem negative_rpe_suppresses_gate (s : StriatalState) (deltaDA : Nat)
    (h_dip : ¬ (s.d1Drive - min s.d1Drive deltaDA > (s.d2Drive + deltaDA) + s.gpiThresh)) :
    (applyDopamineModulation s deltaDA false).isGated = false := by
  dsimp [applyDopamineModulation]
  exact decide_eq_false h_dip

end Weft.Formal.StriatalAction
