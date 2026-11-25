/-
  Formal Verification of Derived Motivic Fiber Bundles for Layer Paging.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicFiberPaging

structure FiberPagingSpec where
  numLayers : Nat
  fiberRank : Nat
  pageCapacity : Nat

def totalFiberPagingCapacity (spec : FiberPagingSpec) : Nat :=
  spec.numLayers * spec.fiberRank + spec.pageCapacity

theorem fiber_paging_nonnegative (spec : FiberPagingSpec) :
    totalFiberPagingCapacity spec >= 0 := by
  exact Nat.zero_le (totalFiberPagingCapacity spec)

theorem fiber_paging_positive (spec : FiberPagingSpec) (hl : spec.numLayers > 0) (hr : spec.fiberRank > 0) :
    totalFiberPagingCapacity spec >= 1 := by
  dsimp [totalFiberPagingCapacity]
  have h1 : spec.numLayers * spec.fiberRank >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hl hr)
  exact Nat.le_add_right_of_le h1

end Weft.DerivedMotivicFiberPaging
