/-
  Formal Verification of Motivic Galois Residual Monodromy Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisResidualMonodromy

structure MonodromyOperator where
  nilpotencyIndex : Nat
  traceNorm : Nat
  isUnipotent : Bool

def maxSpeculativeSteps (O : MonodromyOperator) : Nat :=
  O.nilpotencyIndex + 1

theorem speculative_steps_pos (O : MonodromyOperator) :
    maxSpeculativeSteps O >= 1 := by
  dsimp [maxSpeculativeSteps]
  exact Nat.le_add_left 1 O.nilpotencyIndex

theorem nilpotency_index_bound (O : MonodromyOperator) :
    O.nilpotencyIndex <= maxSpeculativeSteps O := by
  dsimp [maxSpeculativeSteps]
  exact Nat.le_add_right O.nilpotencyIndex 1

end Weft.MotivicGaloisResidualMonodromy
