-- TropicalGrassmannianPruningInvariants.lean
-- Milestone Paper 265: Asymmetric Tropical Grassmannians & Plücker Coordinates
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure PluckerQuad where
  quadId : Nat
  pik : Nat
  pjl : Nat
  pil : Nat
  pjk : Nat
  deriving Repr

def pluckerTerm1 (q : PluckerQuad) : Nat :=
  q.pik + q.pjl

def pluckerTerm2 (q : PluckerQuad) : Nat :=
  q.pil + q.pjk

def tropicalPluckerDiff (q : PluckerQuad) : Nat :=
  let t1 := pluckerTerm1 q
  let t2 := pluckerTerm2 q
  if t1 >= t2 then t1 - t2 else t2 - t1

theorem tropical_plucker_diff_nonneg (q : PluckerQuad) :
    tropicalPluckerDiff q >= 0 := by
  exact Nat.zero_le _

def isHeadTropicallyRedundant (q : PluckerQuad) (epsilon : Nat) : Prop :=
  tropicalPluckerDiff q <= epsilon

theorem redundant_head_bound (q : PluckerQuad) (epsilon : Nat)
    (h_red : isHeadTropicallyRedundant q epsilon) :
    tropicalPluckerDiff q <= epsilon := by
  exact h_red

theorem plucker_diff_symm (t1 t2 : Nat) :
    (if t1 >= t2 then t1 - t2 else t2 - t1) =
    (if t2 >= t1 then t2 - t1 else t1 - t2) := by
  split
  · split <;> omega
  · split <;> omega
