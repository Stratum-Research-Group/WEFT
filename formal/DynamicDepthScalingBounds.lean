/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

namespace WeftFormal.DynamicDepth

/-- Configuration tracking speculative depth bounds. -/
structure DepthBoundsConfig where
  minDepth : Nat
  maxDepth : Nat
  h_order : minDepth <= maxDepth

/-- Clamps a speculative depth value strictly within [minDepth, maxDepth]. -/
def clampDepth (cfg : DepthBoundsConfig) (d : Nat) : Nat :=
  Nat.max cfg.minDepth (Nat.min cfg.maxDepth d)

theorem clamp_ge_min (cfg : DepthBoundsConfig) (d : Nat) :
  clampDepth cfg d >= cfg.minDepth := by
  dsimp [clampDepth]
  exact Nat.le_max_left cfg.minDepth _

theorem clamp_le_max (cfg : DepthBoundsConfig) (d : Nat) :
  clampDepth cfg d <= cfg.maxDepth := by
  dsimp [clampDepth]
  apply Nat.max_le.mpr
  constructor
  · exact cfg.h_order
  · exact Nat.min_le_left cfg.maxDepth d

/-- Direct step depth that always evaluates clampDepth. -/
def stepDepthTarget (cfg : DepthBoundsConfig) (currentDepth : Nat) (alphaScaled : Nat) : Nat :=
  clampDepth cfg (if alphaScaled > 85 then currentDepth + 1 else if alphaScaled < 40 then currentDepth - 2 else currentDepth)

theorem step_target_ge_min (cfg : DepthBoundsConfig) (d : Nat) (alpha : Nat) :
  stepDepthTarget cfg d alpha >= cfg.minDepth :=
  clamp_ge_min cfg _

theorem step_target_le_max (cfg : DepthBoundsConfig) (d : Nat) (alpha : Nat) :
  stepDepthTarget cfg d alpha <= cfg.maxDepth :=
  clamp_le_max cfg _

end WeftFormal.DynamicDepth
