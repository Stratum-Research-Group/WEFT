-- Cuneiform Nucleus (CnF) High-Speed Locomotor Escape & Gallop Bifurcation Invariants
-- Formal verification of CnF escape drive, high-speed gallop transition thresholds,
-- interlimb phase-lag collapse, and sprint cadence bounds.
-- Target #385 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.CnFEscapeLocomotion

/-- State of the mesencephalic cuneiform locomotor escape circuit:
    `cnfDrive`: Cuneiform glutamatergic drive (scaled Nat, 0-100)
    `threatLevel`: Afferent collicular/amygdaloid threat input (scaled Nat, 0-100)
    `gallopThreshold`: Effective drive threshold for asymmetric sprint gallop (scaled Nat, e.g. 45)
    `maxCadenceDeciHz`: Maximum escape cadence limit in tenths of Hz (e.g. 50 for 5.0 Hz)
-/
structure CnFEscapeState where
  cnfDrive : Nat
  threatLevel : Nat
  gallopThreshold : Nat
  maxCadenceDeciHz : Nat
  h_gallop_pos : gallopThreshold > 0
  deriving Repr

/-- Compute effective escape drive:
    (cnfDrive * (30 + (threatLevel * 70) / 100)) / 100. -/
def computeEffectiveDrive (s : CnFEscapeState) : Nat :=
  let threatMod := 30 + (s.threatLevel * 70) / 100
  (s.cnfDrive * threatMod) / 100

/-- Predicate for high-speed gallop/bound regime:
    Effective escape drive strictly exceeds gallopThreshold. -/
def isGallopTransitionActive (s : CnFEscapeState) : Bool :=
  computeEffectiveDrive s >= s.gallopThreshold

/-- Compute interlimb left-right phase lag in degrees:
    Symmetric walk/trot (when effective drive < gallopThreshold): 180 degrees (alternating).
    Asymmetric sprint gallop (when effective drive >= gallopThreshold):
    Phase collapses from 180 toward in-phase synchrony (bounded below by 25 deg). -/
def computeInterlimbPhaseLag (s : CnFEscapeState) : Nat :=
  if isGallopTransitionActive s then
    let excess := computeEffectiveDrive s - s.gallopThreshold
    let collapse := (excess * 155) / 55
    if 180 >= collapse + 25 then
      180 - collapse
    else
      25
  else
    180

/-- Compute sprint escape cadence in tenths of Hz (deci-Hz):
    When stationary (effective drive < 15): 0.
    When trotting (15 <= drive < gallopThreshold): 10 + ((drive - 15) * 15) / 30 (bounded by maxCadenceDeciHz).
    When galloping (drive >= gallopThreshold): 25 + ((drive - gallopThreshold) * 25) / 55 (bounded by maxCadenceDeciHz). -/
def computeEscapeCadence (s : CnFEscapeState) : Nat :=
  let drive := computeEffectiveDrive s
  if drive < 15 then
    0
  else if drive < s.gallopThreshold then
    let raw := 10 + ((drive - 15) * 15) / 30
    if raw <= s.maxCadenceDeciHz then raw else s.maxCadenceDeciHz
  else
    let raw := 25 + ((drive - s.gallopThreshold) * 25) / 55
    if raw <= s.maxCadenceDeciHz then raw else s.maxCadenceDeciHz

/-- Theorem 1: Effective escape drive scales monotonically with cuneiform drive. -/
theorem effective_drive_monotonic_with_cnf (s : CnFEscapeState) (c1 c2 : Nat)
    (h_le : c1 <= c2) :
    let s1 : CnFEscapeState := { s with cnfDrive := c1 }
    let s2 : CnFEscapeState := { s with cnfDrive := c2 }
    computeEffectiveDrive s1 <= computeEffectiveDrive s2 := by
  intro s1 s2
  dsimp [computeEffectiveDrive, s1, s2]
  have h_mul : c1 * (30 + s.threatLevel * 70 / 100) <= c2 * (30 + s.threatLevel * 70 / 100) :=
    Nat.mul_le_mul_right (30 + s.threatLevel * 70 / 100) h_le
  exact Nat.div_le_div_right h_mul

/-- Theorem 2: Interlimb phase lag in sprint gallop is strictly bounded below by 25 degrees. -/
theorem phase_lag_bounded_below (s : CnFEscapeState) :
    25 <= computeInterlimbPhaseLag s := by
  dsimp [computeInterlimbPhaseLag]
  split
  · split
    · omega
    · omega
  · omega

/-- Theorem 3: High-speed escape cadence is strictly bounded by maxCadenceDeciHz. -/
theorem escape_cadence_bounded (s : CnFEscapeState) :
    computeEscapeCadence s <= s.maxCadenceDeciHz := by
  dsimp [computeEscapeCadence]
  split
  · omega
  · split
    · split
      · omega
      · omega
    · split
      · omega
      · omega

/-- Theorem 4: Escalation of cuneiform drive preserves the gallop regime once triggered. -/
theorem cnf_escalation_preserves_gallop (s : CnFEscapeState) (c_new : Nat)
    (h_le : s.cnfDrive <= c_new)
    (h_active : isGallopTransitionActive s = true) :
    isGallopTransitionActive { s with cnfDrive := c_new } = true := by
  dsimp [isGallopTransitionActive] at h_active
  dsimp [isGallopTransitionActive]
  have h_bound : computeEffectiveDrive s >= s.gallopThreshold := of_decide_eq_true h_active
  have h_mono : computeEffectiveDrive s <= computeEffectiveDrive { s with cnfDrive := c_new } :=
    effective_drive_monotonic_with_cnf s s.cnfDrive c_new h_le
  have h_final : computeEffectiveDrive { s with cnfDrive := c_new } >= s.gallopThreshold := by omega
  exact decide_eq_true h_final

end Weft.Formal.CnFEscapeLocomotion
