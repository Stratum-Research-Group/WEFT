/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

namespace WeftFormal

inductive TreeEntropyRegime where
  | Low
  | Medium
  | High
  deriving DecidableEq, Repr

def regimeDepth : TreeEntropyRegime → Nat
  | TreeEntropyRegime.Low    => 8
  | TreeEntropyRegime.Medium => 4
  | TreeEntropyRegime.High   => 2

def regimeBreadth (budget : Nat) : TreeEntropyRegime → Nat
  | TreeEntropyRegime.Low    => 1
  | TreeEntropyRegime.Medium => budget / 4
  | TreeEntropyRegime.High   => budget / 2

theorem low_regime_linear (budget : Nat) :
  regimeBreadth budget TreeEntropyRegime.Low = 1 := rfl

theorem depth_pos (r : TreeEntropyRegime) :
  regimeDepth r > 0 := by
  cases r <;> decide

theorem medium_regime_nodes_bounded (budget : Nat) :
  regimeDepth TreeEntropyRegime.Medium * regimeBreadth budget TreeEntropyRegime.Medium <= budget := by
  dsimp [regimeDepth, regimeBreadth]
  exact Nat.mul_div_le budget 4

theorem high_regime_nodes_bounded (budget : Nat) :
  regimeDepth TreeEntropyRegime.High * regimeBreadth budget TreeEntropyRegime.High <= budget := by
  dsimp [regimeDepth, regimeBreadth]
  exact Nat.mul_div_le budget 2

end WeftFormal
