-- Substantia Nigra Pars Reticulata (SNr) & Motor Thalamus Basal Ganglia Disinhibition Invariants
-- Formal verification of tonic GABAergic output clamping, striatonigral dSPN pause-mediated disinhibition,
-- and low-threshold T-type Ca2+ channel post-inhibitory rebound burst generation.
-- Target #382 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.NigrothalamicGate

/-- Basal ganglia nigrothalamic gate state:
    `dSpnBurstDrive`: Striatonigral dSPN inhibitory burst drive (scaled Nat, 0-100)
    `snrTonicRate`: Baseline tonic GABAergic discharge rate of SNr projection neurons (Hz, scaled Nat)
    `disinhibitionThreshold`: Maximum SNr rate considered an active disinhibitory pause (e.g. 15 Hz)
    `reboundBurstMax`: Peak rebound burst frequency of thalamic relay neurons (e.g. 250 Hz)
-/
structure NigrothalamicGateState where
  dSpnBurstDrive : Nat
  snrTonicRate : Nat
  disinhibitionThreshold : Nat
  reboundBurstMax : Nat
  deriving Repr

/-- Compute effective SNr firing rate under striatonigral inhibition:
    snrTonicRate - (dSpnBurstDrive * snrTonicRate) / 100 (clamped at 0). -/
def computeSNrOutputRate (s : NigrothalamicGateState) : Nat :=
  let inhibition := (s.dSpnBurstDrive * s.snrTonicRate) / 100
  if s.snrTonicRate >= inhibition then
    s.snrTonicRate - inhibition
  else
    0

/-- Predicate for active thalamic disinhibition:
    SNr output is suppressed below disinhibitionThreshold. -/
def isThalamusDisinhibited (s : NigrothalamicGateState) : Bool :=
  computeSNrOutputRate s <= s.disinhibitionThreshold

/-- Compute post-inhibitory rebound burst rate in motor thalamocortical relay cells (VM/VA).
    When disinhibited: (reboundBurstMax * dSpnBurstDrive) / 100. -/
def computeThalamicReboundBurst (s : NigrothalamicGateState) : Nat :=
  if isThalamusDisinhibited s then
    (s.reboundBurstMax * s.dSpnBurstDrive) / 100
  else
    0

/-- Theorem 1: SNr output firing rate decreases monotonically with striatonigral dSPN burst drive. -/
theorem snr_output_decreases_with_dspn (s : NigrothalamicGateState) (d1 d2 : Nat)
    (h_le : d1 <= d2)
    (h_bound : (d2 * s.snrTonicRate) / 100 <= s.snrTonicRate) :
    let s1 : NigrothalamicGateState := { s with dSpnBurstDrive := d1 }
    let s2 : NigrothalamicGateState := { s with dSpnBurstDrive := d2 }
    computeSNrOutputRate s2 <= computeSNrOutputRate s1 := by
  intro s1 s2
  dsimp [computeSNrOutputRate, s1, s2]
  have h_bound1 : (d1 * s.snrTonicRate) / 100 <= s.snrTonicRate := by
    have h_prod : d1 * s.snrTonicRate <= d2 * s.snrTonicRate := Nat.mul_le_mul_right s.snrTonicRate h_le
    have h_div : (d1 * s.snrTonicRate) / 100 <= (d2 * s.snrTonicRate) / 100 := Nat.div_le_div_right h_prod
    omega
  rw [if_pos h_bound1, if_pos h_bound]
  have h_prod : d1 * s.snrTonicRate <= d2 * s.snrTonicRate := Nat.mul_le_mul_right s.snrTonicRate h_le
  have h_div : (d1 * s.snrTonicRate) / 100 <= (d2 * s.snrTonicRate) / 100 := Nat.div_le_div_right h_prod
  omega

/-- Theorem 2: Thalamocortical rebound burst frequency is strictly bounded by reboundBurstMax. -/
theorem rebound_burst_bounded (s : NigrothalamicGateState)
    (h_drive_bound : s.dSpnBurstDrive <= 100) :
    computeThalamicReboundBurst s <= s.reboundBurstMax := by
  dsimp [computeThalamicReboundBurst]
  split
  · have h_mul : s.reboundBurstMax * s.dSpnBurstDrive <= s.reboundBurstMax * 100 :=
      Nat.mul_le_mul_left s.reboundBurstMax h_drive_bound
    have h_div : (s.reboundBurstMax * s.dSpnBurstDrive) / 100 <= (s.reboundBurstMax * 100) / 100 :=
      Nat.div_le_div_right h_mul
    have h_cancel : (s.reboundBurstMax * 100) / 100 = s.reboundBurstMax := by
      exact Nat.mul_div_cancel s.reboundBurstMax (by decide)
    rw [h_cancel] at h_div
    exact h_div
  · omega

/-- Theorem 3: Tonic SNr clamping strictly suppresses thalamocortical rebound burst. -/
theorem snr_tonic_clamp_silences_rebound (s : NigrothalamicGateState)
    (h_clamped : computeSNrOutputRate s > s.disinhibitionThreshold) :
    computeThalamicReboundBurst s = 0 := by
  dsimp [computeThalamicReboundBurst]
  have h_dis : isThalamusDisinhibited s = false := by
    dsimp [isThalamusDisinhibited]
    apply decide_eq_false
    omega
  split
  · rename_i h_true
    rw [h_dis] at h_true
    contradiction
  · rfl

/-- Theorem 4: Striatonigral burst escalation preserves thalamic disinhibition once triggered. -/
theorem dspn_burst_preserves_disinhibition (s : NigrothalamicGateState) (d_new : Nat)
    (h_le : s.dSpnBurstDrive <= d_new)
    (h_bound : (d_new * s.snrTonicRate) / 100 <= s.snrTonicRate)
    (h_dis : isThalamusDisinhibited s = true) :
    isThalamusDisinhibited { s with dSpnBurstDrive := d_new } = true := by
  dsimp [isThalamusDisinhibited] at h_dis
  dsimp [isThalamusDisinhibited]
  have h_val : computeSNrOutputRate s <= s.disinhibitionThreshold := of_decide_eq_true h_dis
  have h_mono : computeSNrOutputRate { s with dSpnBurstDrive := d_new } <= computeSNrOutputRate s :=
    snr_output_decreases_with_dspn s s.dSpnBurstDrive d_new h_le h_bound
  have h_final : computeSNrOutputRate { s with dSpnBurstDrive := d_new } <= s.disinhibitionThreshold :=
    Nat.le_trans h_mono h_val
  exact decide_eq_true h_final

end Weft.Formal.NigrothalamicGate
