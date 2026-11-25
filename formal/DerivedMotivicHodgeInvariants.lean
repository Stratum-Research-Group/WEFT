/-
  Formal Verification of Derived Motivic Hodge Structure Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicHodge

structure MixedHodgeStructure where
  weight : Nat
  h_pq_sum : Nat
  total_dim : Nat
  h_sum_le : h_pq_sum <= total_dim

theorem hodge_filtration_subspace (M : MixedHodgeStructure) :
    M.h_pq_sum <= M.total_dim :=
  M.h_sum_le

def truncationError (total : Nat) (retained : Nat) : Nat :=
  total - retained

theorem truncation_error_bound (total retained : Nat) (_h : retained <= total) :
    truncationError total retained <= total := by
  dsimp [truncationError]
  exact Nat.sub_le total retained

end Weft.DerivedMotivicHodge
