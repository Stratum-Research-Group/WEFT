-- MotivicGaloisAlignmentInvariants.lean
-- Milestone Paper 263: Asymmetric Motivic Galois Groups & Tannakian Categories
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure ModalityToken where
  tokenId : Nat
  modalityId : Nat
  featureNorm : Nat
  deriving Repr

def tannakianBlend (f1 f2 w1 w2 : Nat) : Nat :=
  w1 * f1 + w2 * f2

theorem tannakian_blend_nonneg (f1 f2 w1 w2 : Nat) :
    tannakianBlend f1 f2 w1 w2 >= 0 := by
  exact Nat.zero_le _

theorem tannakian_blend_comm (f1 f2 w : Nat) :
    tannakianBlend f1 f2 w w = tannakianBlend f2 f1 w w := by
  dsimp [tannakianBlend]
  omega

theorem tannakian_blend_mono (f1 f2 f3 w1 w2 : Nat)
    (h_le : f1 <= f2) :
    tannakianBlend f1 f3 w1 w2 <= tannakianBlend f2 f3 w1 w2 := by
  dsimp [tannakianBlend]
  have h_mul : w1 * f1 <= w1 * f2 := Nat.mul_le_mul_left w1 h_le
  omega
