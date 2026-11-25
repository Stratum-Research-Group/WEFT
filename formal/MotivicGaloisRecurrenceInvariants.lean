/-
  Formal Verification of Motivic Galois Recurrence Invariants for 1T Parameter Models.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisRecurrence

structure MotivicRecurrenceSpec where
  numCandidates : Nat
  speculativeDepth : Nat
  acceptanceThreshold : Nat

def totalRecurrenceDegrees (spec : MotivicRecurrenceSpec) : Nat :=
  spec.numCandidates * spec.speculativeDepth + spec.acceptanceThreshold

theorem recurrence_degrees_nonnegative (spec : MotivicRecurrenceSpec) :
    totalRecurrenceDegrees spec >= 0 := by
  exact Nat.zero_le (totalRecurrenceDegrees spec)

theorem recurrence_degrees_positive (spec : MotivicRecurrenceSpec) (hc : spec.numCandidates > 0) (hd : spec.speculativeDepth > 0) :
    totalRecurrenceDegrees spec >= 1 := by
  dsimp [totalRecurrenceDegrees]
  have h1 : spec.numCandidates * spec.speculativeDepth >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hc hd)
  exact Nat.le_add_right_of_le h1

end Weft.MotivicGaloisRecurrence
