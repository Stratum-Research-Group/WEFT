-- MotivicSteenrodRecoveryInvariants.lean
-- Milestone Paper 269: Asymmetric Motivic Cohomological Operations & Steenrod Powers
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure WeightWord where
  wordId : Nat
  rawBits : Nat
  syndromeMask : Nat
  deriving Repr

def xorRecover (bits mask : Nat) : Nat :=
  Nat.xor bits mask

theorem xor_involution (w e : Nat) :
    xorRecover (xorRecover w e) e = w := by
  dsimp [xorRecover]
  rw [Nat.xor_assoc]
  rw [Nat.xor_self]
  rw [Nat.xor_zero]

theorem xor_zero_syndrome (w : Nat) :
    xorRecover w 0 = w := by
  dsimp [xorRecover]
  rw [Nat.xor_zero]

theorem recovered_word_nonneg (bits mask : Nat) :
    xorRecover bits mask >= 0 := by
  exact Nat.zero_le _
