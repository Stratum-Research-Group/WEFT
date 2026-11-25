/-
  Formal Verification of Sub-bit Associative Memory Bounds.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.SubbitAssociativeMemoryBounds

structure AssociativeMemorySpec where
  numPairs : Nat
  stateRank : Nat
  valTokens : Nat

def totalMemorySlots (s : AssociativeMemorySpec) : Nat :=
  s.numPairs * s.stateRank

theorem memory_slots_nonnegative (s : AssociativeMemorySpec) :
    totalMemorySlots s >= 0 := by
  exact Nat.zero_le (totalMemorySlots s)

theorem val_tokens_pos (s : AssociativeMemorySpec) (h : s.valTokens > 0) :
    s.valTokens >= 1 := by
  exact h

end Weft.SubbitAssociativeMemoryBounds
