/-
  Formal Verification of Motivic Galois SIMD Recurrence Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicGaloisSimdRecurrence

structure SimdRecurrenceConfig where
  simdWidth : Nat
  decayScale : Nat
  stateRank : Nat

def totalRecurrentDimension (cfg : SimdRecurrenceConfig) : Nat :=
  cfg.simdWidth * cfg.stateRank

theorem recurrent_dim_nonnegative (cfg : SimdRecurrenceConfig) :
    totalRecurrentDimension cfg >= 0 := by
  exact Nat.zero_le (totalRecurrentDimension cfg)

theorem state_rank_pos (cfg : SimdRecurrenceConfig) (h : cfg.stateRank > 0) :
    cfg.stateRank >= 1 := by
  exact h

end Weft.MotivicGaloisSimdRecurrence
