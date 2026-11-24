-- MotivicSliceQuantizationInvariants.lean
-- Milestone Paper 266: Asymmetric Motivic Spectra & Slice Filtration
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure LayerSlice where
  layerId : Nat
  sliceSensitivity : Nat
  deriving Repr

def allocateSliceBits (bMin alpha sigma : Nat) : Nat :=
  bMin + alpha * sigma

theorem slice_bits_ge_min (bMin alpha sigma : Nat) :
    allocateSliceBits bMin alpha sigma >= bMin := by
  dsimp [allocateSliceBits]
  omega

theorem slice_bits_nonneg (bMin alpha sigma : Nat) :
    allocateSliceBits bMin alpha sigma >= 0 := by
  exact Nat.zero_le _

theorem slice_bits_mono (bMin alpha sigma1 sigma2 : Nat)
    (h_le : sigma1 <= sigma2) :
    allocateSliceBits bMin alpha sigma1 <= allocateSliceBits bMin alpha sigma2 := by
  dsimp [allocateSliceBits]
  have h_mul : alpha * sigma1 <= alpha * sigma2 := Nat.mul_le_mul_left alpha h_le
  omega

theorem slice_budget_additivity (bMin alpha s1 s2 : Nat) :
    allocateSliceBits bMin alpha s1 + allocateSliceBits bMin alpha s2 =
    (2 * bMin) + alpha * (s1 + s2) := by
  dsimp [allocateSliceBits]
  rw [Nat.mul_add alpha s1 s2]
  omega
