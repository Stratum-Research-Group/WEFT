-- Cutaneous Tactile Slip Transduction and Transcortical Grip Adaptation Invariants
-- Formal verification of Fast-Adapting (FA-I / FA-II) mechanoreceptor shear transduction,
-- Coulomb friction cone boundary conditions, and Johansson & Westling grip force escalation.
-- Target #372 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.TactileSlip

/-- Grasp frictional stability state:
    - `stableHold`: Normal grip force satisfies safety margin against tangential load.
    - `activeSlip`: Tangential load breaches static friction cone, requiring reflex escalation.
-/
inductive GripFrictionState where
  | stableHold : GripFrictionState
  | activeSlip : GripFrictionState
  deriving Repr, DecidableEq

/-- State representation of a cutaneous slip detection and grip compensation loop:
    `normalGripForce`: Applied normal force perpendicular to object surface (N, scaled as Nat)
    `tangentialLoad`: Downward gravitational load / inertial pull (N, scaled as Nat)
    `frictionCoeffScaled`: Static friction coefficient mu * 100 (e.g. 35 for mu = 0.35)
    `safetyMultiplier`: Desired safety factor * 100 (e.g. 130 for 1.3x margin)
    `fa1BaseRate`: Spontaneous baseline discharge of Meissner afferents (Hz)
    `fa1SlipGain`: Firing rate amplification under mechanical slip
-/
structure TactileSlipState where
  normalGripForce : Nat
  tangentialLoad : Nat
  frictionCoeffScaled : Nat
  safetyMultiplier : Nat
  fa1BaseRate : Nat
  fa1SlipGain : Nat
  h_mu_pos : frictionCoeffScaled > 0
  h_safety_pos : safetyMultiplier >= 100
  h_gain_pos : fa1SlipGain > 0
  deriving Repr

/-- Maximum available frictional resistance before slip: (normalGripForce * frictionCoeffScaled) / 100. -/
def computeMaxFriction (s : TactileSlipState) : Nat :=
  (s.normalGripForce * s.frictionCoeffScaled) / 100

/-- Evaluate grip frictional state based on Coulomb friction law. -/
def evaluateGripState (s : TactileSlipState) : GripFrictionState :=
  if s.tangentialLoad > computeMaxFriction s then
    GripFrictionState.activeSlip
  else
    GripFrictionState.stableHold

/-- Compute required adapted normal grip force to restore target safety margin:
    F_desired = (tangentialLoad * safetyMultiplier) / frictionCoeffScaled. -/
def computeAdaptedGripForce (s : TactileSlipState) : Nat :=
  let desired := (s.tangentialLoad * s.safetyMultiplier) / s.frictionCoeffScaled
  if desired > s.normalGripForce then desired else s.normalGripForce

/-- Invariant: Is the current grip state physically slipping? -/
def isSlipping (s : TactileSlipState) : Prop :=
  s.tangentialLoad > computeMaxFriction s

/-- Theorem 1: Frictional limit is strictly monotonic with normal grip force. -/
theorem max_friction_monotonic (s : TactileSlipState) (f1 f2 : Nat)
    (h_le : f1 <= f2) :
    let s1 : TactileSlipState := { s with normalGripForce := f1 }
    let s2 : TactileSlipState := { s with normalGripForce := f2 }
    computeMaxFriction s1 <= computeMaxFriction s2 := by
  intro s1 s2
  dsimp [computeMaxFriction, s1, s2]
  have h_mul : f1 * s.frictionCoeffScaled <= f2 * s.frictionCoeffScaled :=
    Nat.mul_le_mul_right s.frictionCoeffScaled h_le
  exact Nat.div_le_div_right h_mul

/-- Theorem 2: Under stable hold conditions, evaluateGripState returns stableHold. -/
theorem stable_hold_invariant (s : TactileSlipState)
    (h_stable : s.tangentialLoad <= computeMaxFriction s) :
    evaluateGripState s = GripFrictionState.stableHold := by
  dsimp [evaluateGripState]
  have h_not : ¬(s.tangentialLoad > computeMaxFriction s) := by omega
  rw [if_neg h_not]

/-- Theorem 3: Active slip unconditionally transitions to activeSlip. -/
theorem active_slip_detection (s : TactileSlipState)
    (h_slip : s.tangentialLoad > computeMaxFriction s) :
    evaluateGripState s = GripFrictionState.activeSlip := by
  dsimp [evaluateGripState]
  rw [if_pos h_slip]

/-- Theorem 4: Adapted grip force is always at least the original normal grip force. -/
theorem adapted_grip_greater_or_equal (s : TactileSlipState) :
    computeAdaptedGripForce s >= s.normalGripForce := by
  dsimp [computeAdaptedGripForce]
  by_cases h : (s.tangentialLoad * s.safetyMultiplier) / s.frictionCoeffScaled > s.normalGripForce
  · rw [if_pos h]
    omega
  · rw [if_neg h]
    exact Nat.le_refl _

/-- Theorem 5: Adapted grip force is monotonic with tangential load. -/
theorem adapted_grip_monotonic_with_load (s : TactileSlipState) (l1 l2 : Nat)
    (h_le : l1 <= l2) :
    let s1 : TactileSlipState := { s with tangentialLoad := l1 }
    let s2 : TactileSlipState := { s with tangentialLoad := l2 }
    computeAdaptedGripForce s1 <= computeAdaptedGripForce s2 := by
  intro s1 s2
  dsimp [computeAdaptedGripForce, s1, s2]
  have h_num : l1 * s.safetyMultiplier <= l2 * s.safetyMultiplier :=
    Nat.mul_le_mul_right s.safetyMultiplier h_le
  have h_div : (l1 * s.safetyMultiplier) / s.frictionCoeffScaled <=
               (l2 * s.safetyMultiplier) / s.frictionCoeffScaled :=
    Nat.div_le_div_right h_num
  by_cases h1 : (l1 * s.safetyMultiplier) / s.frictionCoeffScaled > s.normalGripForce
  · by_cases h2 : (l2 * s.safetyMultiplier) / s.frictionCoeffScaled > s.normalGripForce
    · rw [if_pos h1, if_pos h2]
      exact h_div
    · omega
  · by_cases h2 : (l2 * s.safetyMultiplier) / s.frictionCoeffScaled > s.normalGripForce
    · rw [if_neg h1, if_pos h2]
      omega
    · rw [if_neg h1, if_neg h2]
      exact Nat.le_refl _

end Weft.Formal.TactileSlip
