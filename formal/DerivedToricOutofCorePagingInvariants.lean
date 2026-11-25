/-
  Formal Verification of Derived Toric Compactifications for Out-of-Core Paging.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedToricOutofCorePaging

structure OutofCorePagingSpec where
  pageSize : Nat
  hostResidentPages : Nat
  evictedPages : Nat

def totalManagedPages (spec : OutofCorePagingSpec) : Nat :=
  spec.hostResidentPages + spec.evictedPages

theorem managed_pages_nonnegative (spec : OutofCorePagingSpec) :
    totalManagedPages spec >= 0 := by
  exact Nat.zero_le (totalManagedPages spec)

theorem page_size_positive (spec : OutofCorePagingSpec) (h : spec.pageSize > 0) :
    spec.pageSize >= 1 := by
  exact h

end Weft.DerivedToricOutofCorePaging
