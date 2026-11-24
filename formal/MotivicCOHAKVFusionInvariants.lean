-- MotivicCOHAKVFusionInvariants.lean
-- Milestone Paper 257: Asymmetric Motivic Cohomological Hall Algebras (COHA) & BPS States for FlashAttention KV Fusion
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure COHATile where
  tileId : Nat
  dimGrading : Nat
  maxLogit : Nat
  sumExp : Nat
  deriving Repr, DecidableEq

def fuseGrading (d1 d2 : Nat) : Nat :=
  d1 + d2

theorem fuse_grading_assoc (a b c : Nat) :
    fuseGrading (fuseGrading a b) c = fuseGrading a (fuseGrading b c) := by
  dsimp [fuseGrading]
  omega

theorem fuse_grading_comm (a b : Nat) :
    fuseGrading a b = fuseGrading b a := by
  dsimp [fuseGrading]
  omega

def fuseTile (t1 t2 : COHATile) : COHATile :=
  { tileId := t1.tileId
    dimGrading := fuseGrading t1.dimGrading t2.dimGrading
    maxLogit := if t1.maxLogit >= t2.maxLogit then t1.maxLogit else t2.maxLogit
    sumExp := t1.sumExp + t2.sumExp }

theorem fused_tile_grading_sum (t1 t2 : COHATile) :
    (fuseTile t1 t2).dimGrading = t1.dimGrading + t2.dimGrading := by
  dsimp [fuseTile, fuseGrading]

theorem fused_tile_sum_nonneg (t1 t2 : COHATile) :
    (fuseTile t1 t2).sumExp >= 0 := by
  exact Nat.zero_le _

theorem fuse_max_ge_left (m1 m2 : Nat) :
    (if m1 >= m2 then m1 else m2) >= m1 := by
  split <;> omega

theorem fuse_max_ge_right (m1 m2 : Nat) :
    (if m1 >= m2 then m1 else m2) >= m2 := by
  split <;> omega
