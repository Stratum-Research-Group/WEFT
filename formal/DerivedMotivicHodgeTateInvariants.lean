/-
  Formal Verification of Derived Motivic Hodge-Tate Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicHodgeTate

structure HodgeTateMotive where
  weight : Int
  tateTwist : Int
  extDim : Nat

def isPureTate (M : HodgeTateMotive) : Prop :=
  M.weight = 2 * M.tateTwist

def hodgeFiltrationLength (M : HodgeTateMotive) : Nat :=
  Int.natAbs (M.weight - 2 * M.tateTwist)

theorem pure_tate_filtration_zero (M : HodgeTateMotive) (h : isPureTate M) :
    hodgeFiltrationLength M = 0 := by
  dsimp [isPureTate] at h
  dsimp [hodgeFiltrationLength]
  rw [h]
  simp

theorem ext_dimension_bound (M : HodgeTateMotive) :
    M.extDim >= 0 := by
  exact Nat.zero_le M.extDim

end Weft.DerivedMotivicHodgeTate
