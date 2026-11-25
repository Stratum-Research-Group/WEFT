/-
  Formal Verification of Tropical Toric Stratifications Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TropicalToricStratifications

structure ToricStratification where
  ambientDimension : Nat
  coneDimension : Nat
  h_cone_le : coneDimension <= ambientDimension

def orbitDimension (S : ToricStratification) : Nat :=
  S.ambientDimension - S.coneDimension

theorem orbit_dim_nonneg (S : ToricStratification) :
    orbitDimension S >= 0 := by
  dsimp [orbitDimension]
  exact Nat.zero_le _

theorem orbit_dim_le_ambient (S : ToricStratification) :
    orbitDimension S <= S.ambientDimension := by
  dsimp [orbitDimension]
  exact Nat.sub_le S.ambientDimension S.coneDimension

end Weft.TropicalToricStratifications
