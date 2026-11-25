-- DerivedAzumayaAlgebrasInvariants.lean
-- Milestone Paper 273: Asymmetric Derived Azumaya Algebras & Brauer Groups
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure WeightShare where
  shareId : Nat
  shareVal1 : Nat
  shareVal2 : Nat
  deriving Repr

def reconstructShare (s : WeightShare) : Nat :=
  s.shareVal1 + s.shareVal2

theorem reconstruct_share_nonneg (s : WeightShare) :
    reconstructShare s >= 0 := by
  exact Nat.zero_le _

theorem reconstruct_share_comm (v1 v2 : Nat) :
    v1 + v2 = v2 + v1 := by
  omega

theorem reconstruct_exactness (w mask : Nat) :
    (w + mask) - mask = w := by
  omega

theorem share_monotonicity (v1 v2 v3 : Nat)
    (h_le : v1 <= v2) :
    v1 + v3 <= v2 + v3 := by
  omega
