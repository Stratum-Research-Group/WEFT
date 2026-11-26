-- Cerebellar Flocculus Purkinje Cell Plasticity & VOR Gain Adaptation Invariants
-- Formal verification of climbing fiber retinal slip error signaling,
-- PF-Purkinje synaptic weight decrement (LTD), and monotonic convergence of adapted VOR gain.
-- Target #378 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.CerebellarVORAdaptation

/-- State of cerebellar VOR plasticity circuit:
    `currentGain`: Current ocular reflex gain (scaled Nat, e.g. 100 = 1.0)
    `targetGain`: Required optical gain under lenses (scaled Nat)
    `purkinjeSynapticWeight`: Parallel fiber to Purkinje cell synaptic efficacy (scaled Nat)
    `climbingFiberError`: Inferior olive retinal slip error magnitude (scaled Nat)
    `learningRate`: Adaptation increment (scaled Nat)
-/
structure FloccularPlasticityState where
  currentGain : Nat
  targetGain : Nat
  purkinjeSynapticWeight : Nat
  climbingFiberError : Nat
  learningRate : Nat
  deriving Repr

/-- Compute absolute retinal slip error under head rotation speed (normalized to 100):
    |targetGain - currentGain|. -/
def computeSlipError (s : FloccularPlasticityState) : Nat :=
  if s.targetGain >= s.currentGain then
    s.targetGain - s.currentGain
  else
    s.currentGain - s.targetGain

/-- Adaptive weight update under climbing fiber error:
    In Purkinje cells, climbing fiber discharge induces LTD (downregulation of PF-Purkinje weight),
    which disinhibits vestibular nucleus output and increases VOR gain. -/
def applyPlasticityStep (s : FloccularPlasticityState) : FloccularPlasticityState :=
  if s.currentGain < s.targetGain then
    let gainIncrement := s.learningRate
    let newGain := s.currentGain + gainIncrement
    let weightDecrement := if s.purkinjeSynapticWeight >= s.learningRate then s.learningRate else s.purkinjeSynapticWeight
    { s with
      currentGain := if newGain <= s.targetGain then newGain else s.targetGain,
      purkinjeSynapticWeight := s.purkinjeSynapticWeight - weightDecrement,
      climbingFiberError := if s.targetGain >= newGain then s.targetGain - newGain else 0 }
  else
    s

/-- Theorem 1: Under-calibrated VOR gain increases strictly monotonically during adaptation step. -/
theorem gain_adaptation_monotonic (s : FloccularPlasticityState)
    (h_under : s.currentGain < s.targetGain)
    (h_rate_pos : s.learningRate > 0) :
    s.currentGain < (applyPlasticityStep s).currentGain := by
  dsimp [applyPlasticityStep]
  rw [if_pos h_under]
  dsimp
  split
  · omega
  · omega

/-- Theorem 2: Parallel fiber-Purkinje synaptic weight decreases monotonically (LTD) during gain escalation. -/
theorem purkinje_weight_ltd_monotonic (s : FloccularPlasticityState)
    (h_under : s.currentGain < s.targetGain) :
    (applyPlasticityStep s).purkinjeSynapticWeight <= s.purkinjeSynapticWeight := by
  dsimp [applyPlasticityStep]
  rw [if_pos h_under]
  dsimp
  split
  · omega
  · omega

/-- Theorem 3: Retinal slip error decreases monotonically after adaptive step. -/
theorem slip_error_decreases (s : FloccularPlasticityState)
    (h_under : s.currentGain < s.targetGain)
    (h_step_bound : s.currentGain + s.learningRate <= s.targetGain) :
    computeSlipError (applyPlasticityStep s) <= computeSlipError s := by
  dsimp [computeSlipError, applyPlasticityStep]
  rw [if_pos h_under]
  dsimp
  have h_le_target : s.currentGain + s.learningRate <= s.targetGain := h_step_bound
  rw [if_pos h_le_target]
  have h_under1 : s.targetGain >= s.currentGain := by omega
  have h_under2 : s.targetGain >= s.currentGain + s.learningRate := h_step_bound
  rw [if_pos h_under1, if_pos h_under2]
  omega

/-- Theorem 4: Target reach invariant: Once currentGain matches targetGain, slip error is strictly zero. -/
theorem target_reached_zero_slip (s : FloccularPlasticityState)
    (h_eq : s.currentGain = s.targetGain) :
    computeSlipError s = 0 := by
  dsimp [computeSlipError]
  rw [h_eq]
  simp

end Weft.Formal.CerebellarVORAdaptation
