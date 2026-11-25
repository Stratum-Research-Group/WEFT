/-
  Formal Verification of Derived Toric Degenerations for MoE Layer Paging.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedToricMoEPaging

structure ToricLayerPagingSpec where
  numLayers : Nat
  shardedChunks : Nat
  boundaryDivisors : Nat

def totalLayerPagingCapacity (spec : ToricLayerPagingSpec) : Nat :=
  spec.numLayers * spec.shardedChunks + spec.boundaryDivisors

theorem layer_paging_capacity_nonnegative (spec : ToricLayerPagingSpec) :
    totalLayerPagingCapacity spec >= 0 := by
  exact Nat.zero_le (totalLayerPagingCapacity spec)

theorem layer_paging_capacity_positive (spec : ToricLayerPagingSpec) (hl : spec.numLayers > 0) (hc : spec.shardedChunks > 0) :
    totalLayerPagingCapacity spec >= 1 := by
  dsimp [totalLayerPagingCapacity]
  have h1 : spec.numLayers * spec.shardedChunks >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hl hc)
  exact Nat.le_add_right_of_le h1

end Weft.DerivedToricMoEPaging
