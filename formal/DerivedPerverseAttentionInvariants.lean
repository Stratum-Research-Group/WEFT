-- DerivedPerverseAttentionInvariants.lean
-- Milestone Paper 255: Asymmetric Derived Intersection Homology & Perverse Sheaf Micro-Local Attention Routing
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure StratumToken where
  tokenId : Nat
  codimension : Nat
  deriving Repr, DecidableEq

def middlePerversity (k : Nat) : Nat :=
  if k >= 2 then (k - 2) / 2 else 0

theorem middle_perversity_base : middlePerversity 2 = 0 := by
  dsimp [middlePerversity]

theorem middle_perversity_nonneg (k : Nat) : middlePerversity k >= 0 := by
  exact Nat.zero_le _

def stratumCodimDiff (s1 s2 : StratumToken) : Nat :=
  if s1.codimension >= s2.codimension then
    s1.codimension - s2.codimension
  else
    s2.codimension - s1.codimension

theorem stratum_codim_diff_symm (s1 s2 : StratumToken) :
    stratumCodimDiff s1 s2 = stratumCodimDiff s2 s1 := by
  dsimp [stratumCodimDiff]
  by_cases h : s1.codimension >= s2.codimension
  · by_cases h_eq : s1.codimension = s2.codimension
    · simp [h_eq]
    · have h_lt : ¬ (s2.codimension >= s1.codimension) := by omega
      simp [h, h_lt]
  · have h_gt : s2.codimension >= s1.codimension := by omega
    simp [h, h_gt]

def perverseAttentionPenalty (s1 s2 : StratumToken) (delta : Nat) : Nat :=
  let diff := stratumCodimDiff s1 s2
  middlePerversity (diff + 2) * delta

theorem perverse_penalty_symm (s1 s2 : StratumToken) (delta : Nat) :
    perverseAttentionPenalty s1 s2 delta = perverseAttentionPenalty s2 s1 delta := by
  dsimp [perverseAttentionPenalty]
  rw [stratum_codim_diff_symm s1 s2]

theorem perverse_penalty_same_stratum (s : StratumToken) (delta : Nat) :
    perverseAttentionPenalty s s delta = 0 := by
  dsimp [perverseAttentionPenalty, stratumCodimDiff, middlePerversity]
  split
  · have h_sub : s.codimension - s.codimension = 0 := by omega
    rw [h_sub]
    dsimp
    exact Nat.zero_mul delta
  · have h_sub : s.codimension - s.codimension = 0 := by omega
    rw [h_sub]
    dsimp
    exact Nat.zero_mul delta
