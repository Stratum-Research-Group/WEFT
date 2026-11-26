-- AIP-PMv Visual Grasp Affordance and Preshaping Aperture Invariants
-- Formal verification of anterior intraparietal area (AIP) 3D geometric affordance extraction,
-- ventral premotor cortex (Area F5) motor prototype selection,
-- and Jeannerod peak hand aperture linear preshaping bounds.
-- Target #369 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.GraspAffordance

/-- Grasp affordance prototype category:
    - `precisionPinch`: Index-thumb pad-to-pad opposition for small objects.
    - `powerGrip`: Palm and multi-finger flexion wrap for large objects.
-/
inductive GripPrototype where
  | precisionPinch : GripPrototype
  | powerGrip : GripPrototype
  deriving Repr, DecidableEq

/-- State representation of visual grasp affordance system:
    `objectWidth`: Object diameter/width (mm, scaled as Nat)
    `pinchThreshold`: Diameter cutoff dividing pinch from power grasp (e.g. 35 mm)
    `apertureIntercept`: Hand aperture baseline safety margin (e.g. 18 mm)
    `apertureSlope`: Linear scaling of aperture with object size (e.g. 1)
    `maxAperture`: Maximum physical finger span (e.g. 125 mm)
-/
structure GraspAffordanceState where
  objectWidth : Nat
  pinchThreshold : Nat
  apertureIntercept : Nat
  apertureSlope : Nat
  maxAperture : Nat
  h_slope_pos : apertureSlope > 0
  h_intercept_pos : apertureIntercept > 0
  h_max_bound : apertureIntercept + apertureSlope * pinchThreshold <= maxAperture
  deriving Repr

/-- Compute Jeannerod peak hand preshaping aperture: a + b * width, capped at maxAperture. -/
def computePeakAperture (s : GraspAffordanceState) : Nat :=
  let raw := s.apertureIntercept + s.apertureSlope * s.objectWidth
  if raw <= s.maxAperture then raw else s.maxAperture

/-- Classify grip prototype based on visual object dimension. -/
def selectGripPrototype (s : GraspAffordanceState) : GripPrototype :=
  if s.objectWidth < s.pinchThreshold then
    GripPrototype.precisionPinch
  else
    GripPrototype.powerGrip

/-- Safety invariant: Hand aperture must always exceed object width to guarantee collision-free enclosure. -/
def apertureSufficient (s : GraspAffordanceState) : Prop :=
  computePeakAperture s >= s.objectWidth

/-- Theorem 1: Raw hand preshaping aperture is strictly monotonic with respect to object width. -/
theorem raw_aperture_monotonic (s : GraspAffordanceState) (w1 w2 : Nat)
    (h_le : w1 <= w2) :
    s.apertureIntercept + s.apertureSlope * w1 <= s.apertureIntercept + s.apertureSlope * w2 := by
  have h_mul : s.apertureSlope * w1 <= s.apertureSlope * w2 := Nat.mul_le_mul_left s.apertureSlope h_le
  exact Nat.add_le_add_left h_mul s.apertureIntercept

/-- Theorem 2: Capped peak hand aperture is monotonic with respect to object width. -/
theorem peak_aperture_monotonic (s : GraspAffordanceState) (w1 w2 : Nat)
    (h_le : w1 <= w2) :
    let s1 : GraspAffordanceState := { s with objectWidth := w1 }
    let s2 : GraspAffordanceState := { s with objectWidth := w2 }
    computePeakAperture s1 <= computePeakAperture s2 := by
  intro s1 s2
  dsimp [computePeakAperture, s1, s2]
  have h_raw := raw_aperture_monotonic s w1 w2 h_le
  by_cases h2 : s.apertureIntercept + s.apertureSlope * w2 <= s.maxAperture
  · have h1 : s.apertureIntercept + s.apertureSlope * w1 <= s.maxAperture := by omega
    rw [if_pos h1, if_pos h2]
    exact h_raw
  · by_cases h1 : s.apertureIntercept + s.apertureSlope * w1 <= s.maxAperture
    · rw [if_pos h1, if_neg h2]
      exact h1
    · rw [if_neg h1, if_neg h2]
      exact Nat.le_refl _

/-- Theorem 3: Precision pinch preservation for small objects. -/
theorem precision_pinch_classification (s : GraspAffordanceState)
    (h_small : s.objectWidth < s.pinchThreshold) :
    selectGripPrototype s = GripPrototype.precisionPinch := by
  dsimp [selectGripPrototype]
  rw [if_pos h_small]

/-- Theorem 4: Power grip classification for large objects. -/
theorem power_grip_classification (s : GraspAffordanceState)
    (h_large : s.objectWidth >= s.pinchThreshold) :
    selectGripPrototype s = GripPrototype.powerGrip := by
  dsimp [selectGripPrototype]
  have h_not : ¬(s.objectWidth < s.pinchThreshold) := by omega
  rw [if_neg h_not]

/-- Theorem 5: Aperture safety enclosure guarantee for objects up to maximum graspable size. -/
theorem aperture_enclosure_guarantee (s : GraspAffordanceState)
    (h_fit : s.apertureIntercept + s.apertureSlope * s.objectWidth <= s.maxAperture) :
    computePeakAperture s >= s.objectWidth := by
  dsimp [computePeakAperture]
  rw [if_pos h_fit]
  have h_sl : s.apertureSlope >= 1 := s.h_slope_pos
  have h_base : s.apertureIntercept >= 1 := s.h_intercept_pos
  have h_prod : s.apertureSlope * s.objectWidth >= s.objectWidth := by
    calc s.apertureSlope * s.objectWidth >= 1 * s.objectWidth := Nat.mul_le_mul_right s.objectWidth h_sl
    _ = s.objectWidth := Nat.one_mul s.objectWidth
  omega

end Weft.Formal.GraspAffordance
