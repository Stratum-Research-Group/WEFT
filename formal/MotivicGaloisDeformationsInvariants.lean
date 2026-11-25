/-
  Formal Verification of Motivic Galois Deformation Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisDeformations

structure DeformationData where
  tangentDim : Nat
  obstructionDim : Nat

def maxObstructionBound (d : Nat) : Nat :=
  (d * (d - 1)) / 2

theorem obstruction_dim_nonneg (d : Nat) :
    maxObstructionBound d >= 0 := by
  dsimp [maxObstructionBound]
  exact Nat.zero_le _

theorem obstruction_dim_bound (d : Nat) :
    maxObstructionBound d <= d * d := by
  dsimp [maxObstructionBound]
  have h1 : d * (d - 1) <= d * d := by
    apply Nat.mul_le_mul_left
    exact Nat.sub_le d 1
  exact Nat.le_trans (Nat.div_le_self (d * (d - 1)) 2) h1

end Weft.MotivicGaloisDeformations
