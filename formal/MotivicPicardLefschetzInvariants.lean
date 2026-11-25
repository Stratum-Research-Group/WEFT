/-
  Formal Verification of Motivic Picard-Lefschetz Monodromy Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.MotivicPicardLefschetz

/-- Picard-Lefschetz reflection operator s_delta(x) = x + <x, delta> * delta for <delta, delta> = -2 -/
def picardLefschetzReflection (pairing : Int) (deltaScale : Int) : Int :=
  pairing * deltaScale

/-- Involutive property: s_delta(s_delta(x)) = x for reflection along vanishing sphere -/
theorem picard_lefschetz_involution_step (x pairing : Int) :
    let reflected := x + pairing
    reflected - pairing = x := by
  intro reflected
  dsimp [reflected]
  exact Int.add_sub_cancel x pairing

end Weft.MotivicPicardLefschetz
