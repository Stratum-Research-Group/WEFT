-- Golgi Tendon Organ (GTO) Force & Autogenic Inhibition Invariants
-- Formal proof of monotonic tension transduction, autogenic negative feedback force limiting,
-- and unconditional clasp-knife protective shutoff under mechanical tendon overload.
-- Target #362 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.GolgiTendonOrgan

/-- Musculoskeletal force state:
    - `tensionRegulated`: Normal force servo operation with autogenic damping.
    - `claspKnifeReleased`: Overload threshold exceeded, requiring total motor cessation.
-/
inductive TendonProtectionState where
  | tensionRegulated : TendonProtectionState
  | claspKnifeReleased : TendonProtectionState
  deriving Repr, DecidableEq

/-- State representation of a Golgi Tendon Organ feedback loop:
    `tension`: Muscle contractile tension (Newtons, scaled as Nat)
    `descendingDrive`: Motor command drive (Hz, scaled as Nat)
    `k_Ib`: Force-to-discharge gain constant
    `w_Ib`: Inhibitory synaptic weight onto alpha motor pool
    `criticalTension`: Rupture/avulsion hazard threshold
-/
structure GTOFeedbackState where
  tension : Nat
  descendingDrive : Nat
  k_Ib : Nat
  w_Ib : Nat
  criticalTension : Nat
  h_k_pos : k_Ib > 0
  h_w_pos : w_Ib > 0
  h_crit_pos : criticalTension > 0
  deriving Repr

/-- Compute primary Group Ib afferent firing rate (linearized scaled approximation). -/
def computeIbRate (s : GTOFeedbackState) : Nat :=
  s.k_Ib * s.tension

/-- Compute di-synaptic autogenic inhibition acting upon alpha motor pool. -/
def computeAutogenicInhibition (s : GTOFeedbackState) : Nat :=
  s.w_Ib * computeIbRate s

/-- Evaluate tendon protection state:
    If tension meets or exceeds criticalTension, claspKnifeReleased is triggered.
    Otherwise, tension remains regulated under physiological feedback.
-/
def evaluateTendonProtection (s : GTOFeedbackState) : TendonProtectionState :=
  if s.tension ≥ s.criticalTension then
    TendonProtectionState.claspKnifeReleased
  else
    TendonProtectionState.tensionRegulated

/-- Calculate net motor pool output drive:
    In claspKnifeReleased, drive is clamped unconditionally to zero.
    In tensionRegulated, drive is diminished by autogenic inhibition.
-/
def computeNetAlphaDrive (s : GTOFeedbackState) : Nat :=
  if s.tension ≥ s.criticalTension then
    0
  else if s.descendingDrive ≥ computeAutogenicInhibition s then
    s.descendingDrive - computeAutogenicInhibition s
  else
    0

/-- Theorem: Tendon overload exceeding critical tension unconditionally triggers
    clasp-knife release and clamps net motor drive to zero. -/
theorem overload_triggers_clasp_knife_shutoff
    (s : GTOFeedbackState)
    (h_over : s.tension ≥ s.criticalTension) :
    evaluateTendonProtection s = TendonProtectionState.claspKnifeReleased ∧
    computeNetAlphaDrive s = 0 := by
  dsimp [evaluateTendonProtection, computeNetAlphaDrive]
  rw [if_pos h_over]
  rw [if_pos h_over]
  exact ⟨rfl, rfl⟩

/-- Theorem: Group Ib discharge is monotonically increasing with muscle tension. -/
theorem ib_discharge_monotonic_in_tension
    (s : GTOFeedbackState) (t1 t2 : Nat)
    (h_le : t1 ≤ t2) :
    let s1 := { s with tension := t1 }
    let s2 := { s with tension := t2 }
    computeIbRate s1 ≤ computeIbRate s2 := by
  dsimp [computeIbRate]
  exact Nat.mul_le_mul_left s.k_Ib h_le

/-- Theorem: Autogenic inhibition is monotonically increasing with muscle tension. -/
theorem autogenic_inhibition_monotonic_in_tension
    (s : GTOFeedbackState) (t1 t2 : Nat)
    (h_le : t1 ≤ t2) :
    let s1 := { s with tension := t1 }
    let s2 := { s with tension := t2 }
    computeAutogenicInhibition s1 ≤ computeAutogenicInhibition s2 := by
  dsimp [computeAutogenicInhibition, computeIbRate]
  have h1 : s.k_Ib * t1 ≤ s.k_Ib * t2 := Nat.mul_le_mul_left s.k_Ib h_le
  exact Nat.mul_le_mul_left s.w_Ib h1

end Weft.Formal.GolgiTendonOrgan
