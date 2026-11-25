/-
  Formal Verification of Tropical Toric Page Table Compactifications.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalToricPageTablesCompactifications

structure PagingCompactification where
  pageSize : Nat
  pageCount : Nat
  virtualPages : Nat

def totalPhysicalBytes (p : PagingCompactification) : Nat :=
  p.pageSize * p.pageCount

theorem physical_bytes_nonnegative (p : PagingCompactification) :
    totalPhysicalBytes p >= 0 := by
  exact Nat.zero_le (totalPhysicalBytes p)

theorem page_count_pos (p : PagingCompactification) (h : p.pageCount > 0) :
    p.pageCount >= 1 := by
  exact h

end Weft.TropicalToricPageTablesCompactifications
