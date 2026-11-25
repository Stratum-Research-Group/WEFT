/-
Copyright (c) 2026 Antigravity Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Antigravity Team
-/

set_option linter.unusedVariables false

namespace WeftFormal.ActiveInference

/-- The 6 stages of the cog2 closed-loop cognitive cycle. -/
inductive CognitiveStage where
  | Ingestion : CognitiveStage
  | SurpriseFiltering : CognitiveStage
  | AttentionalIgnition : CognitiveStage
  | GlobalBroadcast : CognitiveStage
  | MetacognitiveInterrogation : CognitiveStage
  | TemporalUpdate : CognitiveStage
  deriving DecidableEq, Repr

def stageRank : CognitiveStage → Nat
  | CognitiveStage.Ingestion => 0
  | CognitiveStage.SurpriseFiltering => 1
  | CognitiveStage.AttentionalIgnition => 2
  | CognitiveStage.GlobalBroadcast => 3
  | CognitiveStage.MetacognitiveInterrogation => 4
  | CognitiveStage.TemporalUpdate => 5

def nextStage : CognitiveStage → CognitiveStage
  | CognitiveStage.Ingestion => CognitiveStage.SurpriseFiltering
  | CognitiveStage.SurpriseFiltering => CognitiveStage.AttentionalIgnition
  | CognitiveStage.AttentionalIgnition => CognitiveStage.GlobalBroadcast
  | CognitiveStage.GlobalBroadcast => CognitiveStage.MetacognitiveInterrogation
  | CognitiveStage.MetacognitiveInterrogation => CognitiveStage.TemporalUpdate
  | CognitiveStage.TemporalUpdate => CognitiveStage.Ingestion

theorem forward_stage_strictly_increasing (s : CognitiveStage) (h : s ≠ CognitiveStage.TemporalUpdate) :
  stageRank s < stageRank (nextStage s) := by
  cases s
  · decide
  · decide
  · decide
  · decide
  · decide
  · contradiction

/-- Free-energy state under Bayesian surprise minimization. -/
structure InferenceState where
  step : Nat
  freeEnergyScaled : Nat
  surpriseThreshold : Nat

/-- Closed-loop error suppression step: F_{t+1} <= F_t -/
def suppressPredictionError (st : InferenceState) : InferenceState :=
  { st with 
    step := st.step + 1,
    freeEnergyScaled := st.freeEnergyScaled / 2 }

theorem error_suppression_monotonic (st : InferenceState) :
  (suppressPredictionError st).freeEnergyScaled <= st.freeEnergyScaled := by
  dsimp [suppressPredictionError]
  exact Nat.div_le_self st.freeEnergyScaled 2

theorem step_advances (st : InferenceState) :
  (suppressPredictionError st).step = st.step + 1 := rfl

end WeftFormal.ActiveInference
