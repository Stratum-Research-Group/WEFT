/-
  Ternary dot-product bound: a theorem that quantifies over actual weight and
  activation vectors (not bare Nat arithmetic).

  For any ternary weight vector w (every entry in {-1, 0, +1}) and any integer
  activation vector x, |<w, x>| <= sum |x_i|. This is the exact accumulator
  bound that the 1.58-bit GEMV kernels rely on (no overflow beyond the L1 norm
  of the activation).

  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.TernaryDot

/-- Integer dot product over lists (truncates to the shorter list). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- L1 norm of an integer vector. -/
def l1 (x : List Int) : Nat := (x.map Int.natAbs).sum

/-- A weight vector is ternary when every entry has absolute value at most 1. -/
def IsTernary (w : List Int) : Prop := ∀ t ∈ w, t.natAbs ≤ 1

theorem ternary_dot_bound :
    ∀ (w x : List Int), IsTernary w → (dot w x).natAbs ≤ l1 x
  | [], x, _ => by simp [dot, l1]
  | _ :: _, [], _ => by simp [dot, l1]
  | a :: as, b :: bs, h => by
      have ha : a.natAbs ≤ 1 := h a (List.mem_cons_self ..)
      have hrest : IsTernary as := fun t ht => h t (List.mem_cons_of_mem _ ht)
      have ih := ternary_dot_bound as bs hrest
      have hmul : (a * b).natAbs ≤ b.natAbs := by
        rw [Int.natAbs_mul]
        calc a.natAbs * b.natAbs ≤ 1 * b.natAbs := Nat.mul_le_mul_right _ ha
          _ = b.natAbs := Nat.one_mul _
      have hadd := Int.natAbs_add_le (a * b) (dot as bs)
      simp only [dot, l1, List.map_cons, List.sum_cons] at *
      omega

end Weft.TernaryDot
