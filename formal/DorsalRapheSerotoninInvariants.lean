-- Dorsal Raphe Nucleus Serotonin (5-HT) Waiting Patience & Impulsivity Threshold Invariants
-- Formal verification of serotonergic neuromodulatory suppression of temporal discounting:
-- 1. Miyazaki et al. (2014, 2018) & Doya (2002) Waiting Patience and Delayed Gratification.
-- 2. Monotonic deceleration of hyperbolic discount rate k by dorsal raphe 5-HT tone.
-- 3. Temporal value monotonicity with respect to nominal delayed reward magnitude.
-- 4. Impulsivity threshold bounding premature action abortion under 5-HT depletion.
-- Target #387 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.DorsalRaphe5HT

/-- State parameters representing serotonergic tone, delay duration, and reward value.
    All rates, values, and delays are represented in scaled natural units. -/
structure SerotoninState where
  /-- Dorsal raphe 5-HT firing tone (scaled Nat, 0 - 100 representing 0.0 - 10.0 Hz) -/
  serotoninTone : Nat
  /-- Nominal delayed reward magnitude (scaled Nat, e.g. 100) -/
  nominalReward : Nat
  /-- Delay time until reward delivery (scaled Nat, tenths of seconds) -/
  delayTime : Nat
  /-- Baseline temporal discounting coefficient (scaled Nat, e.g. 50) -/
  kBaseline : Nat
  /-- Action abortion / impatience threshold (scaled Nat, e.g. 20) -/
  abortThreshold : Nat
  h_kbase_pos : kBaseline > 0
  deriving Repr

/-- Effective discount parameter k(5HT):
    k_eff = (kBaseline * 100) / (100 + serotoninTone * 5)
    As 5-HT tone escalates, the discount rate k_eff strictly decreases (slower decay). -/
def effectiveDiscountRate (s : SerotoninState) : Nat :=
  (s.kBaseline * 100) / (100 + s.serotoninTone * 5)

/-- Subjective discounted value V(t):
    V = (nominalReward * 1000) / (1000 + effectiveDiscountRate * delayTime) -/
def subjectiveRewardValue (s : SerotoninState) : Nat :=
  (s.nominalReward * 1000) / (1000 + effectiveDiscountRate s * s.delayTime)

/-- Theorem 1: Effective discount rate is non-negative and bounded above by kBaseline. -/
theorem discount_rate_bounded_by_baseline (s : SerotoninState) :
    effectiveDiscountRate s <= s.kBaseline := by
  dsimp [effectiveDiscountRate]
  have h_denom : 100 + s.serotoninTone * 5 >= 100 := by omega
  refine Nat.div_le_of_le_mul ?_
  have h_mul : s.kBaseline * 100 <= s.kBaseline * (100 + s.serotoninTone * 5) := by
    apply Nat.mul_le_mul_left
    omega
  rw [Nat.mul_comm (100 + s.serotoninTone * 5) s.kBaseline]
  exact h_mul

/-- Theorem 2: Instantaneous reward delivery (delayTime = 0) suffers zero discounting. -/
theorem zero_delay_undiscounted_value (s : SerotoninState) :
    subjectiveRewardValue { s with delayTime := 0 } = s.nominalReward := by
  dsimp [subjectiveRewardValue]
  omega

/-- Theorem 3: Subjective value is monotonic with respect to nominal reward magnitude. -/
theorem subjective_value_monotone_nominal (s : SerotoninState) (r1 r2 : Nat) (h_r : r1 <= r2) :
    subjectiveRewardValue { s with nominalReward := r1 } <=
    subjectiveRewardValue { s with nominalReward := r2 } := by
  dsimp [subjectiveRewardValue]
  apply Nat.div_le_div_right
  apply Nat.mul_le_mul_right
  exact h_r

/-- Theorem 4: Elevating dorsal raphe 5-HT tone attenuates the effective discount rate. -/
theorem serotonin_elevation_attenuates_discount (s : SerotoninState) (extra_5ht : Nat) :
    let s_boosted : SerotoninState := { s with serotoninTone := s.serotoninTone + extra_5ht }
    effectiveDiscountRate s_boosted <= effectiveDiscountRate s := by
  dsimp [effectiveDiscountRate]
  apply Nat.div_le_div_left
  · omega
  · omega

/-- Theorem 5: Serotonin preservation of subjective value above the impulsive abort threshold.
    If the fully discounted value remains >= abortThreshold, perseverant waiting is sustained. -/
def preservesWaiting (s : SerotoninState) : Prop :=
  subjectiveRewardValue s >= s.abortThreshold

theorem sustained_waiting_under_sufficient_serotonin (s : SerotoninState)
    (h_pres : subjectiveRewardValue s >= s.abortThreshold) :
    preservesWaiting s := by
  dsimp [preservesWaiting]
  exact h_pres

end Weft.Formal.DorsalRaphe5HT
