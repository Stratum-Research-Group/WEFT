-- Vestibulo-Ocular Reflex (VOR) Three-Neuron Arc & Retinal Slip Invariants
-- Formal verification of semicircular canal velocity transduction, compensatory counter-rotation,
-- unity gain maintenance, and retinal slip bounding.
-- Target #377 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.VORRetinalSlip

/-- State of the Vestibulo-Ocular Reflex arc:
    `headVelocity`: Angular velocity of the head in yaw (scaled as Nat, e.g. tenths of deg/s)
    `vorGainNumerator`: Gain scaling numerator (e.g. 100 for 1.0)
    `vorGainDenominator`: Gain scaling denominator (e.g. 100)
    `maxSlipThreshold`: Maximum allowable retinal slip before visual degradation (scaled Nat)
-/
structure VORState where
  headVelocity : Nat
  vorGainNumerator : Nat
  vorGainDenominator : Nat
  maxSlipThreshold : Nat
  h_denom_pos : vorGainDenominator > 0
  deriving Repr

/-- Compute absolute compensatory eye velocity:
    (headVelocity * vorGainNumerator) / vorGainDenominator. -/
def computeEyeVelocity (s : VORState) : Nat :=
  (s.headVelocity * s.vorGainNumerator) / s.vorGainDenominator

/-- Compute residual retinal slip magnitude when gain is under-compensating (gain <= 1.0):
    headVelocity - eyeVelocity. -/
def computeRetinalSlip (s : VORState) : Nat :=
  if s.headVelocity >= computeEyeVelocity s then
    s.headVelocity - computeEyeVelocity s
  else
    0

/-- Predicate for dynamic visual acuity preservation:
    Retinal slip remains bounded under maxSlipThreshold. -/
def isSlipBounded (s : VORState) : Bool :=
  computeRetinalSlip s <= s.maxSlipThreshold

/-- Theorem 1: Compensatory eye velocity is monotonic with head rotation velocity. -/
theorem eye_velocity_monotonic_with_head (s : VORState) (v1 v2 : Nat)
    (h_le : v1 <= v2) :
    let s1 : VORState := { s with headVelocity := v1 }
    let s2 : VORState := { s with headVelocity := v2 }
    computeEyeVelocity s1 <= computeEyeVelocity s2 := by
  intro s1 s2
  dsimp [computeEyeVelocity, s1, s2]
  have h_mul : v1 * s.vorGainNumerator <= v2 * s.vorGainNumerator := Nat.mul_le_mul_right s.vorGainNumerator h_le
  exact Nat.div_le_div_right h_mul

/-- Theorem 2: Ideal unity gain eliminates retinal slip:
    When vorGainNumerator = vorGainDenominator, retinal slip is strictly zero. -/
theorem unity_gain_zero_slip (s : VORState)
    (h_unity : s.vorGainNumerator = s.vorGainDenominator) :
    computeRetinalSlip s = 0 := by
  have h_eye : computeEyeVelocity s = s.headVelocity := by
    dsimp [computeEyeVelocity]
    rw [h_unity]
    rw [Nat.mul_comm, Nat.mul_div_right s.headVelocity s.h_denom_pos]
  dsimp [computeRetinalSlip]
  rw [h_eye]
  simp

/-- Theorem 3: Increased head velocity increases eye velocity compensation. -/
theorem head_velocity_compensation_bound (s : VORState) (delta : Nat) :
    let s_faster : VORState := { s with headVelocity := s.headVelocity + delta }
    computeEyeVelocity s <= computeEyeVelocity s_faster := by
  intro s_faster
  dsimp [computeEyeVelocity, s_faster]
  have h_add : s.headVelocity * s.vorGainNumerator <= (s.headVelocity + delta) * s.vorGainNumerator := by
    apply Nat.mul_le_mul_right
    omega
  exact Nat.div_le_div_right h_add

/-- Theorem 4: Retinal slip is strictly non-negative and zero under perfect match. -/
theorem slip_nonnegative_and_bounded (s : VORState)
    (h_match : computeEyeVelocity s = s.headVelocity) :
    computeRetinalSlip s = 0 ∧ isSlipBounded s = true := by
  have h_slip : computeRetinalSlip s = 0 := by
    dsimp [computeRetinalSlip]
    rw [h_match]
    simp
  have h_bound : isSlipBounded s = true := by
    dsimp [isSlipBounded]
    rw [h_slip]
    exact decide_eq_true (Nat.zero_le s.maxSlipThreshold)
  exact ⟨h_slip, h_bound⟩

end Weft.Formal.VORRetinalSlip
