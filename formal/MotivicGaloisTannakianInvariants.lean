/-
  Formal Verification of Motivic Galois Tannakian Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisTannakian

structure TannakianGroupScheme where
  maximalTorusRank : Nat
  rootCount : Nat

def characterValuation (weight : Nat) (characterDeg : Nat) : Nat :=
  weight * characterDeg

theorem character_valuation_zero_at_zero (w : Nat) :
    characterValuation w 0 = 0 := by
  dsimp [characterValuation]

theorem character_valuation_le (w deg : Nat) :
    w <= characterValuation w deg + w := by
  dsimp [characterValuation]
  exact Nat.le_add_left w (w * deg)

end Weft.MotivicGaloisTannakian
