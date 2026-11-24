-- DerivedCriticalLociInvariants.lean
-- Milestone Paper 261: Asymmetric Derived Critical Loci & Vanishing Cycles
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure CriticalGradient where
  paramId : Nat
  rawGradNorm : Nat
  vanishingCyclePenalty : Nat
  h_bound : vanishingCyclePenalty <= rawGradNorm
  deriving Repr

def regularizedGradNorm (g : CriticalGradient) : Nat :=
  g.rawGradNorm - g.vanishingCyclePenalty

theorem regularized_grad_nonneg (g : CriticalGradient) :
    regularizedGradNorm g >= 0 := by
  exact Nat.zero_le _

theorem regularized_grad_le_raw (g : CriticalGradient) :
    regularizedGradNorm g <= g.rawGradNorm := by
  dsimp [regularizedGradNorm]
  omega

theorem regularized_grad_monotonic (g1 g2 : CriticalGradient)
    (h_le : g1.rawGradNorm <= g2.rawGradNorm)
    (h_pen : g1.vanishingCyclePenalty = g2.vanishingCyclePenalty) :
    regularizedGradNorm g1 <= regularizedGradNorm g2 := by
  dsimp [regularizedGradNorm]
  rw [h_pen]
  omega
