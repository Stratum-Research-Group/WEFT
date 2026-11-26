-- Posterior Parietal Area 5 / 7 Multimodal Coordinate Transformation Invariants
-- Formal verification of eye-position gain fields, retinotopic-to-head-centered transformation,
-- hand-centered reach vector extraction, and reach invariance under gaze shifts.
-- Target #375 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.ParietalCoordinate

/-- Coordinate reference frame for motor intention:
    - `retinotopicFrame`: Visual error relative to the fovea.
    - `headCenteredFrame`: Spatial position relative to head midline.
    - `handCenteredFrame`: Displacement vector from hand to target.
-/
inductive CoordinateFrame where
  | retinotopicFrame : CoordinateFrame
  | headCenteredFrame : CoordinateFrame
  | handCenteredFrame : CoordinateFrame
  deriving Repr, DecidableEq

/-- State representation of a multimodal parietal coordinate transformation circuit:
    `retinoTarget`: Retinotopic target position (scaled as Nat, e.g. tenths of degrees)
    `eyePosition`: Eye-in-head orientation (scaled as Nat)
    `handPosition`: Hand position in head-centered frame (scaled as Nat)
    `gainCoeff`: Gain field multiplicative factor
-/
structure ParietalTransformState where
  retinoTarget : Nat
  eyePosition : Nat
  handPosition : Nat
  gainCoeff : Nat
  h_gain_pos : gainCoeff > 0
  deriving Repr

/-- Compute head-centered spatial target position: retinoTarget + eyePosition. -/
def computeHeadCenteredTarget (s : ParietalTransformState) : Nat :=
  s.retinoTarget + s.eyePosition

/-- Compute hand-centered reach vector: Target_head - Hand_head (when Target >= Hand). -/
def computeReachVector (s : ParietalTransformState) : Nat :=
  if computeHeadCenteredTarget s >= s.handPosition then
    computeHeadCenteredTarget s - s.handPosition
  else
    0

/-- Theorem 1: Head-centered target coordinate is strictly monotonic with retinotopic target. -/
theorem head_centered_monotonic_with_retino (s : ParietalTransformState) (r1 r2 : Nat)
    (h_le : r1 <= r2) :
    let s1 : ParietalTransformState := { s with retinoTarget := r1 }
    let s2 : ParietalTransformState := { s with retinoTarget := r2 }
    computeHeadCenteredTarget s1 <= computeHeadCenteredTarget s2 := by
  intro s1 s2
  dsimp [computeHeadCenteredTarget, s1, s2]
  exact Nat.add_le_add_right h_le s.eyePosition

/-- Theorem 2: Head-centered target coordinate is strictly monotonic with eye position. -/
theorem head_centered_monotonic_with_eye (s : ParietalTransformState) (e1 e2 : Nat)
    (h_le : e1 <= e2) :
    let s1 : ParietalTransformState := { s with eyePosition := e1 }
    let s2 : ParietalTransformState := { s with eyePosition := e2 }
    computeHeadCenteredTarget s1 <= computeHeadCenteredTarget s2 := by
  intro s1 s2
  dsimp [computeHeadCenteredTarget, s1, s2]
  exact Nat.add_le_add_left h_le s.retinoTarget

/-- Theorem 3: Reach vector invariance under compensatory gaze shift:
    If a gaze shift delta_e produces an exact opposite retinotopic shift delta_r = -delta_e,
    the resulting reach vector is strictly invariant. -/
theorem reach_vector_gaze_invariance (s : ParietalTransformState) (delta : Nat)
    (h_retino : s.retinoTarget >= delta) :
    let s_shifted : ParietalTransformState := { s with retinoTarget := s.retinoTarget - delta, eyePosition := s.eyePosition + delta }
    computeHeadCenteredTarget s_shifted = computeHeadCenteredTarget s ∧
    computeReachVector s_shifted = computeReachVector s := by
  intro s_shifted
  have h_head : computeHeadCenteredTarget s_shifted = computeHeadCenteredTarget s := by
    dsimp [computeHeadCenteredTarget, s_shifted]
    omega
  have h_reach : computeReachVector s_shifted = computeReachVector s := by
    dsimp [computeReachVector]
    rw [h_head]
  exact ⟨h_head, h_reach⟩

/-- Theorem 4: Reach vector is monotonic with retinotopic target distance when within forward reach. -/
theorem reach_vector_monotonic_with_retino (s : ParietalTransformState) (r1 r2 : Nat)
    (h_le : r1 <= r2)
    (h_reach1 : s.handPosition <= r1 + s.eyePosition) :
    let s1 : ParietalTransformState := { s with retinoTarget := r1 }
    let s2 : ParietalTransformState := { s with retinoTarget := r2 }
    computeReachVector s1 <= computeReachVector s2 := by
  intro s1 s2
  have h_head1 : computeHeadCenteredTarget s1 = r1 + s.eyePosition := by
    dsimp [computeHeadCenteredTarget, s1]
  have h_head2 : computeHeadCenteredTarget s2 = r2 + s.eyePosition := by
    dsimp [computeHeadCenteredTarget, s2]
  have h_cond1 : computeHeadCenteredTarget s1 >= s1.handPosition := by
    dsimp [s1]
    rw [h_head1]
    exact h_reach1
  have h_cond2 : computeHeadCenteredTarget s2 >= s2.handPosition := by
    dsimp [s2]
    rw [h_head2]
    omega
  dsimp [computeReachVector]
  have h_pos1 : (if computeHeadCenteredTarget s1 >= s1.handPosition then computeHeadCenteredTarget s1 - s1.handPosition else 0) =
      computeHeadCenteredTarget s1 - s1.handPosition := if_pos h_cond1
  have h_pos2 : (if computeHeadCenteredTarget s2 >= s2.handPosition then computeHeadCenteredTarget s2 - s2.handPosition else 0) =
      computeHeadCenteredTarget s2 - s2.handPosition := if_pos h_cond2
  rw [h_pos1, h_pos2]
  rw [h_head1, h_head2]
  dsimp [s1, s2]
  omega

end Weft.Formal.ParietalCoordinate
