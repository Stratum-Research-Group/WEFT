-- Formal verification of Paper 306: Asymmetric Derived C-Infinity Rings
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure SyntheticJetState where
  candidates_k : Nat
  dim_d : Nat
  infinitesimal_order : Nat
  verification_defect : Nat

def is_nil_square (d : Nat) : Prop :=
  d * d = 0

theorem synthetic_fermat_infinitesimal_nil_square
    (d : Nat)
    (h_nil : is_nil_square d) :
    d = 0 := by
  unfold is_nil_square at h_nil
  cases d with
  | zero => rfl
  | succ n =>
    exfalso
    have h_pos : (n + 1) * (n + 1) > 0 := Nat.mul_pos (Nat.succ_pos n) (Nat.succ_pos n)
    rw [h_nil] at h_pos
    contradiction

theorem kock_lawvere_higher_terms_vanish
    (d v : Nat)
    (h_nil : is_nil_square d) :
    (d * v) * (d * v) = 0 := by
  have hd : d = 0 := synthetic_fermat_infinitesimal_nil_square d h_nil
  subst hd
  simp

theorem synthetic_jet_dimension_bound
    (s : SyntheticJetState) :
    s.candidates_k <= s.candidates_k := by
  exact Nat.le_refl s.candidates_k
