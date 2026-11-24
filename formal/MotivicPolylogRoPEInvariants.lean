-- MotivicPolylogRoPEInvariants.lean
-- Milestone Paper 260: Asymmetric Motivic Polylogarithms & Zagier Conjectures
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure RoPEPhase where
  dimId : Nat
  quantizedAngle : Nat
  regulatorCorrection : Nat
  h_corr : regulatorCorrection <= quantizedAngle
  deriving Repr

def dequantizeAngle (p : RoPEPhase) : Nat :=
  p.quantizedAngle - p.regulatorCorrection

theorem dequantized_angle_nonneg (p : RoPEPhase) :
    dequantizeAngle p >= 0 := by
  exact Nat.zero_le _

theorem dequantized_angle_le_quantized (p : RoPEPhase) :
    dequantizeAngle p <= p.quantizedAngle := by
  dsimp [dequantizeAngle]
  omega

theorem dequantize_angle_mono (p1 p2 : RoPEPhase)
    (h_le : p1.quantizedAngle <= p2.quantizedAngle)
    (h_corr_eq : p1.regulatorCorrection = p2.regulatorCorrection) :
    dequantizeAngle p1 <= dequantizeAngle p2 := by
  dsimp [dequantizeAngle]
  rw [h_corr_eq]
  omega
