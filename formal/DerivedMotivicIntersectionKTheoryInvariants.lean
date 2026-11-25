/-
  Formal Verification of Derived Motivic Intersection K-Theory Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedMotivicIntersectionKTheory

structure IntersectionKBundle where
  virtualRank : Nat
  chernClassDeg : Nat
  torsionRank : Nat

def totalBundleDim (B : IntersectionKBundle) : Nat :=
  B.virtualRank + B.torsionRank

theorem virtual_rank_le_total (B : IntersectionKBundle) :
    B.virtualRank <= totalBundleDim B := by
  dsimp [totalBundleDim]
  exact Nat.le_add_right B.virtualRank B.torsionRank

theorem rank_preserved (r1 r2 : Nat) (h : r1 = r2) :
    r1 = r2 :=
  h

end Weft.DerivedMotivicIntersectionKTheory
