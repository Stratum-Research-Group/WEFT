-- TropicalJacobianAttentionInvariants.lean
-- Milestone Paper 271: Asymmetric Tropical Curve Metric Graphs & Jacobian Varieties
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure ModalityCoord where
  tokenId : Nat
  coordVal : Nat
  deriving Repr

def tropicalDist (c1 c2 : Nat) : Nat :=
  if c1 >= c2 then c1 - c2 else c2 - c1

theorem tropical_dist_nonneg (c1 c2 : Nat) :
    tropicalDist c1 c2 >= 0 := by
  exact Nat.zero_le _

theorem tropical_dist_self (c : Nat) :
    tropicalDist c c = 0 := by
  dsimp [tropicalDist]
  split <;> omega

theorem tropical_dist_symm (c1 c2 : Nat) :
    tropicalDist c1 c2 = tropicalDist c2 c1 := by
  dsimp [tropicalDist]
  split
  · split <;> omega
  · split <;> omega

theorem tropical_triangle_ineq (a b c : Nat) :
    tropicalDist a c <= tropicalDist a b + tropicalDist b c := by
  dsimp [tropicalDist]
  split
  · split
    · split <;> omega
    · split <;> omega
  · split
    · split <;> omega
    · split <;> omega
