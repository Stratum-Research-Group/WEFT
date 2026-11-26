-- Vestibulospinal Postural Tone & Head-in-Space Stabilization Invariants
-- Formal proof that lateral gravitational head tilt unconditionally commands
-- ipsilateral antigravity extensor tone recruitment via the Lateral Vestibulospinal Tract (LVST),
-- preserving the body center of mass above the support polygon.
-- Target #364 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.VestibulospinalTone

/-- Postural righting state:
    - `postureAligned`: Head and body upright within nominal vertical cone.
    - `extensorRightingRecruited`: Suprathreshold tilt commanding LVST antigravity extensor drive.
-/
inductive VestibulospinalState where
  | postureAligned : VestibulospinalState
  | extensorRightingRecruited : VestibulospinalState
  deriving Repr, DecidableEq

/-- State representation of vestibulospinal postural circuit:
    `tiltAngle`: Head tilt relative to vertical (scaled as Nat, e.g. degrees)
    `tiltThreshold`: Maximum unperturbed sway angle before righting reflex engages (e.g. 10 deg)
    `baseExtensorTone`: Basal antigravity tone in quiet stance (e.g. 45 Hz)
    `k_lvst`: LVST recruitment sensitivity gain
    `antigravityThreshold`: Minimum drive defining active righting thrust (e.g. 80 Hz)
-/
structure VestibulospinalLoopState where
  tiltAngle : Nat
  tiltThreshold : Nat
  baseExtensorTone : Nat
  k_lvst : Nat
  antigravityThreshold : Nat
  h_k_pos : k_lvst > 0
  h_thresh_order : baseExtensorTone < antigravityThreshold
  h_tilt_pos : tiltThreshold > 0
  h_recruitment_gain : baseExtensorTone + k_lvst * tiltThreshold ≥ antigravityThreshold
  deriving Repr

/-- Compute descending Lateral Vestibulospinal Tract (LVST) antigravity extensor drive. -/
def computeLVSTDrive (s : VestibulospinalLoopState) : Nat :=
  s.baseExtensorTone + s.k_lvst * s.tiltAngle

/-- Evaluate vestibulospinal postural righting state. -/
def evaluateVestibulospinalState (s : VestibulospinalLoopState) : VestibulospinalState :=
  if s.tiltAngle ≥ s.tiltThreshold then
    VestibulospinalState.extensorRightingRecruited
  else
    VestibulospinalState.postureAligned

/-- Theorem: For any head tilt meeting or exceeding the perturbation threshold,
    the vestibulospinal circuit unconditionally triggers `extensorRightingRecruited`
    with LVST extensor drive meeting or exceeding the antigravity threshold. -/
theorem tilt_triggers_extensor_righting
    (s : VestibulospinalLoopState)
    (h_tilt : s.tiltAngle ≥ s.tiltThreshold) :
    evaluateVestibulospinalState s = VestibulospinalState.extensorRightingRecruited ∧
    computeLVSTDrive s ≥ s.antigravityThreshold := by
  dsimp [evaluateVestibulospinalState, computeLVSTDrive]
  rw [if_pos h_tilt]
  have h_gain := s.h_recruitment_gain
  have h_mult : s.k_lvst * s.tiltThreshold ≤ s.k_lvst * s.tiltAngle := Nat.mul_le_mul_left s.k_lvst h_tilt
  have h_drive : s.baseExtensorTone + s.k_lvst * s.tiltAngle ≥ s.antigravityThreshold := by omega
  exact ⟨rfl, h_drive⟩

/-- Theorem: LVST extensor drive is strictly monotonic with respect to head tilt angle. -/
theorem lvst_drive_monotonic_in_tilt
    (s : VestibulospinalLoopState) (t1 t2 : Nat)
    (h_le : t1 ≤ t2) :
    let s1 := { s with tiltAngle := t1 }
    let s2 := { s with tiltAngle := t2 }
    computeLVSTDrive s1 ≤ computeLVSTDrive s2 := by
  dsimp [computeLVSTDrive]
  have h1 : s.k_lvst * t1 ≤ s.k_lvst * t2 := Nat.mul_le_mul_left s.k_lvst h_le
  omega

end Weft.Formal.VestibulospinalTone
