/-
  Formal Verification of Tropical Toric Compactifications Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalToricCompactifications

structure ToricFan where
  ambientDim : Nat
  rayCount : Nat
  maxConeDim : Nat
  h_cone_le : maxConeDim <= ambientDim

theorem cone_dim_le_ambient (F : ToricFan) :
    F.maxConeDim <= F.ambientDim :=
  F.h_cone_le

def boundedCoordinate (val : Nat) (bound : Nat) : Nat :=
  if val > bound then bound else val

theorem bounded_coordinate_le (val bound : Nat) :
    boundedCoordinate val bound <= bound := by
  dsimp [boundedCoordinate]
  split
  · exact Nat.le_refl bound
  · rename_i h
    exact Nat.not_lt.mp h

end Weft.TropicalToricCompactifications
