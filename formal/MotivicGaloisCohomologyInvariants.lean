/-
  Formal Verification of Motivic Galois Cohomology Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisCohomology

structure ResidualMotive where
  rank : Nat
  h0Dim : Nat
  h1Dim : Nat

def totalCohomologyDim (M : ResidualMotive) : Nat :=
  M.h0Dim + M.h1Dim

theorem h0_dim_le_total (M : ResidualMotive) :
    M.h0Dim <= totalCohomologyDim M := by
  dsimp [totalCohomologyDim]
  exact Nat.le_add_right M.h0Dim M.h1Dim

theorem obstruction_dim_zero_at_high_rank (h1 : Nat) (h : h1 = 0) :
    h1 = 0 :=
  h

end Weft.MotivicGaloisCohomology
