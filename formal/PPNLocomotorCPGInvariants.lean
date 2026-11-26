-- Pedunculopontine Nucleus (PPN) Subpopulations & Locomotor CPG Phase Gating Invariants
-- Formal verification of glutamatergic (vGluT2) stepping initiation, cholinergic (ChAT) posture tone,
-- basal ganglia GABAergic clamping, and spinal central pattern generator (CPG) frequency scaling.
-- Target #384 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.PPNLocomotorCPG

/-- State of the mesencephalic locomotor region and spinal CPG:
    `pprnGlutDrive`: PPN glutamatergic descending drive (scaled Nat, 0-100)
    `bgGabaClamp`: Basal ganglia inhibitory output (SNr/GPi) onto PPN (scaled Nat, 0-100)
    `initiationThreshold`: Minimum net drive required to break stance (scaled Nat, e.g. 15)
    `maxCadenceDeciHz`: Upper biophysical limit on stepping cadence in tenths of Hz (e.g. 28 for 2.8 Hz)
-/
structure LocomotorCPGState where
  pprnGlutDrive : Nat
  bgGabaClamp : Nat
  initiationThreshold : Nat
  maxCadenceDeciHz : Nat
  h_thresh_pos : initiationThreshold > 0
  deriving Repr

/-- Compute net descending locomotor drive:
    pprnGlutDrive - bgGabaClamp (when pprnGlutDrive >= bgGabaClamp). -/
def computeNetLocomotorDrive (s : LocomotorCPGState) : Nat :=
  if s.pprnGlutDrive >= s.bgGabaClamp then
    s.pprnGlutDrive - s.bgGabaClamp
  else
    0

/-- Predicate for active locomotion:
    Net drive strictly exceeds initiationThreshold. -/
def isLocomotionInitiated (s : LocomotorCPGState) : Bool :=
  computeNetLocomotorDrive s >= s.initiationThreshold

/-- Compute stepping cadence in tenths of Hz (deci-Hz):
    When stationary: 0.
    When active: 8 (0.8 Hz) + (netDrive - initiationThreshold) * 20 / 85 (capped at maxCadenceDeciHz). -/
def computeSteppingCadence (s : LocomotorCPGState) : Nat :=
  if isLocomotionInitiated s then
    let net := computeNetLocomotorDrive s
    let raw := 8 + ((net - s.initiationThreshold) * 20) / 85
    if raw <= s.maxCadenceDeciHz then raw else s.maxCadenceDeciHz
  else
    0

/-- Theorem 1: Net locomotor drive increases monotonically with PPN glutamatergic drive. -/
theorem net_drive_monotonic_with_ppn (s : LocomotorCPGState) (g1 g2 : Nat)
    (h_le : g1 <= g2)
    (h_clamp : s.bgGabaClamp <= g1) :
    let s1 : LocomotorCPGState := { s with pprnGlutDrive := g1 }
    let s2 : LocomotorCPGState := { s with pprnGlutDrive := g2 }
    computeNetLocomotorDrive s1 <= computeNetLocomotorDrive s2 := by
  intro s1 s2
  dsimp [computeNetLocomotorDrive, s1, s2]
  have h1 : g1 >= s.bgGabaClamp := h_clamp
  have h2 : g2 >= s.bgGabaClamp := by omega
  rw [if_pos h1, if_pos h2]
  omega

/-- Theorem 2: Basal ganglia GABAergic clamping strictly halts locomotion when exceeding drive. -/
theorem bg_clamp_enforces_arrest (s : LocomotorCPGState)
    (h_clamp : s.bgGabaClamp >= s.pprnGlutDrive) :
    computeSteppingCadence s = 0 := by
  dsimp [computeSteppingCadence]
  have h_init : isLocomotionInitiated s = false := by
    dsimp [isLocomotionInitiated, computeNetLocomotorDrive]
    split
    · have h_eq : s.pprnGlutDrive = s.bgGabaClamp := by omega
      rw [h_eq]
      apply decide_eq_false
      have h_pos := s.h_thresh_pos
      omega
    · apply decide_eq_false
      have h_pos := s.h_thresh_pos
      omega
  split
  · rename_i h_true
    rw [h_init] at h_true
    contradiction
  · rfl

/-- Theorem 3: Stepping cadence is strictly bounded by maxCadenceDeciHz. -/
theorem stepping_cadence_bounded (s : LocomotorCPGState) :
    computeSteppingCadence s <= s.maxCadenceDeciHz := by
  dsimp [computeSteppingCadence]
  split
  · split
    · omega
    · omega
  · omega

/-- Theorem 4: Elevating PPN drive preserves active locomotion once initiated. -/
theorem ppn_drive_preserves_locomotion (s : LocomotorCPGState) (g_new : Nat)
    (h_le : s.pprnGlutDrive <= g_new)
    (h_init : isLocomotionInitiated s = true) :
    isLocomotionInitiated { s with pprnGlutDrive := g_new } = true := by
  dsimp [isLocomotionInitiated, computeNetLocomotorDrive] at h_init
  dsimp [isLocomotionInitiated, computeNetLocomotorDrive]
  split at h_init
  · have h_bound : s.pprnGlutDrive - s.bgGabaClamp >= s.initiationThreshold := of_decide_eq_true h_init
    have h_new_ge : g_new >= s.bgGabaClamp := by omega
    split
    · have h_final : g_new - s.bgGabaClamp >= s.initiationThreshold := by omega
      exact decide_eq_true h_final
    · omega
  · have h_not : ¬(0 >= s.initiationThreshold) := by
      have h_pos := s.h_thresh_pos
      omega
    have h_false : decide (0 >= s.initiationThreshold) = false := decide_eq_false h_not
    rw [h_false] at h_init
    contradiction

end Weft.Formal.PPNLocomotorCPG
