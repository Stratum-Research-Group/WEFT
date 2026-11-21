/-!
  # Formal Verification of Packed Bitboard Equivalence in Lean 4

  This module proves that the bitboard unpacking operator:
  s(b) = if b then 1 else -1
  is an exact isomorphism between Boolean coordinates and bipolar signs {-1, +1}.
  Consequently:
  1. No sign information is lost during binary bitboard compression.
  2. The unpacking map is invertible on the bipolar domain.
-/

-- Definition 1: Unpack boolean bit into bipolar integer {-1, +1}
def unpack_bit (b : Bool) : Int :=
  if b = true then 1 else -1

-- Definition 2: Pack bipolar integer {-1, +1} back into boolean bit
def pack_bit (s : Int) : Bool :=
  s > 0

-- Theorem 1: Left Invertibility on Bipolar Domain
theorem unpack_pack_id (s : Int) (h : s = 1 ∨ s = -1) :
  unpack_bit (pack_bit s) = s := by
  cases h with
  | inl h1 =>
    rw [h1]
    rfl
  | inr h2 =>
    rw [h2]
    rfl

-- Theorem 2: Right Invertibility on Boolean Domain
theorem pack_unpack_id (b : Bool) :
  pack_bit (unpack_bit b) = b := by
  cases b <;> rfl

-- Theorem 3: Injectivity of Unpack
theorem unpack_injective :
  Function.Injective unpack_bit := by
  intro x y h_eq
  cases x <;> cases y <;> revert h_eq <;> decide
