/-
  Formal Verification of Derived Symplectic KV State Rollback Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedSymplecticKVStateRollback

structure RollbackSpec where
  numLayers : Nat
  stateRank : Nat
  checkpointCapacity : Nat

def totalRollbackCapacity (spec : RollbackSpec) : Nat :=
  spec.numLayers * spec.stateRank + spec.checkpointCapacity

theorem rollback_capacity_nonnegative (spec : RollbackSpec) :
    totalRollbackCapacity spec >= 0 := by
  exact Nat.zero_le (totalRollbackCapacity spec)

theorem rollback_capacity_positive (spec : RollbackSpec) (hl : spec.numLayers > 0) (hr : spec.stateRank > 0) :
    totalRollbackCapacity spec >= 1 := by
  dsimp [totalRollbackCapacity]
  have h1 : spec.numLayers * spec.stateRank >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hl hr)
  exact Nat.le_add_right_of_le h1

end Weft.DerivedSymplecticKVStateRollback
