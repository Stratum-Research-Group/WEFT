/-
  Formal Verification of Derived Motivic Galois Group Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicGalois

/-- Unipotent radical dimension bound for rank-r factorized tensor motives. -/
def unipotentRadicalMaxDim (r : Nat) : Nat :=
  (r * (r - 1)) / 2

theorem unipotent_radical_dim_nonneg (r : Nat) :
    unipotentRadicalMaxDim r >= 0 := by
  dsimp [unipotentRadicalMaxDim]
  exact Nat.zero_le _

theorem unipotent_radical_dim_zero_at_one :
    unipotentRadicalMaxDim 1 = 0 := by
  rfl

theorem unipotent_radical_dim_bound (r : Nat) :
    unipotentRadicalMaxDim r <= r * r := by
  dsimp [unipotentRadicalMaxDim]
  have h1 : r * (r - 1) <= r * r := by
    apply Nat.mul_le_mul_left
    exact Nat.sub_le r 1
  exact Nat.le_trans (Nat.div_le_self (r * (r - 1)) 2) h1

end Weft.DerivedMotivicGalois
