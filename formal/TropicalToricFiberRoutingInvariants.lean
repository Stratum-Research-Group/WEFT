/-
  Formal Verification of Tropical Toric Fiber Routing Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalToricFiberRouting

structure ToricFiberRoutingSpec where
  numExperts : Nat
  topK : Nat
  sharedBias : Nat

def totalFiberRoutingDim (spec : ToricFiberRoutingSpec) : Nat :=
  spec.numExperts * spec.topK + spec.sharedBias

theorem fiber_routing_nonnegative (spec : ToricFiberRoutingSpec) :
    totalFiberRoutingDim spec >= 0 := by
  exact Nat.zero_le (totalFiberRoutingDim spec)

theorem fiber_routing_positive (spec : ToricFiberRoutingSpec) (he : spec.numExperts > 0) (hk : spec.topK > 0) :
    totalFiberRoutingDim spec >= 1 := by
  dsimp [totalFiberRoutingDim]
  have h1 : spec.numExperts * spec.topK >= 1 := Nat.succ_le_of_lt (Nat.mul_pos he hk)
  exact Nat.le_add_right_of_le h1

end Weft.TropicalToricFiberRouting
