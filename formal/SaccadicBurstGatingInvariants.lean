-- Superior Colliculus Deep Layers (SCdl) Motor Map & Saccadic Burst Generator (PPRF / RIP) Invariants
-- Formal verification of Omnipause Neuron (OPN) tonic clamping, burst neuron activation,
-- main-sequence duration scaling, and asymptotic peak velocity saturation bounds.
-- Target #379 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.SaccadicBurstGating

/-- Saccadic motor generation state:
    `targetAmplitude`: Desired saccadic gaze displacement in degrees (scaled Nat, e.g. tenths of deg)
    `opnTonicDrive`: OPN tonic clamping inhibition (scaled Nat, 0-100)
    `scCollicularDrive`: Superior colliculus population motor vector drive (scaled Nat, 0-100)
    `baseDuration`: Fixed base duration overhead (scaled Nat, e.g. 20 ms)
    `durationSlope`: Scaling of duration with amplitude (scaled Nat, e.g. 2 ms per deg)
-/
structure SaccadicGeneratorState where
  targetAmplitude : Nat
  opnTonicDrive : Nat
  scCollicularDrive : Nat
  baseDuration : Nat
  durationSlope : Nat
  deriving Repr

/-- Predicate for OPN pause / disinhibition:
    OPNs are suppressed when superior colliculus drive strictly exceeds tonic clamp. -/
def isOPNPaused (s : SaccadicGeneratorState) : Bool :=
  s.scCollicularDrive > s.opnTonicDrive

/-- Compute saccadic burst duration in milliseconds:
    baseDuration + (targetAmplitude * durationSlope). -/
def computeSaccadeDuration (s : SaccadicGeneratorState) : Nat :=
  s.baseDuration + s.targetAmplitude * s.durationSlope

/-- Compute PPRF burst firing rate when un-gated (0 if clamped by OPN).
    When active: 100 + targetAmplitude * 15 (saturating at 850 spikes/s). -/
def computePPRFBurstRate (s : SaccadicGeneratorState) : Nat :=
  if isOPNPaused s then
    let raw := 100 + s.targetAmplitude * 15
    if raw <= 850 then raw else 850
  else
    0

/-- Theorem 1: Saccadic duration scales monotonically with target amplitude. -/
theorem saccade_duration_monotonic (s : SaccadicGeneratorState) (a1 a2 : Nat)
    (h_le : a1 <= a2) :
    let s1 : SaccadicGeneratorState := { s with targetAmplitude := a1 }
    let s2 : SaccadicGeneratorState := { s with targetAmplitude := a2 }
    computeSaccadeDuration s1 <= computeSaccadeDuration s2 := by
  intro s1 s2
  dsimp [computeSaccadeDuration, s1, s2]
  have h_mul : a1 * s.durationSlope <= a2 * s.durationSlope := Nat.mul_le_mul_right s.durationSlope h_le
  exact Nat.add_le_add_left h_mul s.baseDuration

/-- Theorem 2: PPRF burst neuron firing rate is strictly bounded by saturation ceiling (850 Hz). -/
theorem pprf_burst_rate_bounded (s : SaccadicGeneratorState) :
    computePPRFBurstRate s <= 850 := by
  dsimp [computePPRFBurstRate]
  split
  · split
    · omega
    · omega
  · omega

/-- Theorem 3: OPN tonic clamp strictly silences PPRF burst activity. -/
theorem opn_clamp_enforces_silence (s : SaccadicGeneratorState)
    (h_clamped : s.scCollicularDrive <= s.opnTonicDrive) :
    computePPRFBurstRate s = 0 := by
  dsimp [computePPRFBurstRate]
  have h_paused : isOPNPaused s = false := by
    dsimp [isOPNPaused]
    apply decide_eq_false
    omega
  split
  · rename_i h_true
    rw [h_paused] at h_true
    contradiction
  · rfl

/-- Theorem 4: Collicular drive escalation preserves OPN disinhibition once active. -/
theorem collicular_drive_preserves_pause (s : SaccadicGeneratorState) (drive_new : Nat)
    (h_active : isOPNPaused s = true)
    (h_inc : s.scCollicularDrive <= drive_new) :
    isOPNPaused { s with scCollicularDrive := drive_new } = true := by
  dsimp [isOPNPaused] at h_active
  dsimp [isOPNPaused]
  have h_gt : s.scCollicularDrive > s.opnTonicDrive := of_decide_eq_true h_active
  have h_gt2 : drive_new > s.opnTonicDrive := by omega
  exact decide_eq_true h_gt2

end Weft.Formal.SaccadicBurstGating
