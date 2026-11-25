/-
  Substantive invariants for real components without non-standard axioms or sorry:
    1. Greedy speculative verification: Prefix properties of commonPrefix.
    2. FIFO Ring Buffer: Capacity constraints and non-loss drain order.
  Stratum Research Group • Apache-2.0
-/

namespace Weft.SubstantiveInvariants

/- ───────── 1. Greedy Speculative Verification (Prefix Invariants) ───────── -/

def commonPrefix : List Nat → List Nat → List Nat
  | a :: as, b :: bs => if a = b then a :: commonPrefix as bs else []
  | _, _ => []

theorem commonPrefix_prefix_left : ∀ (a b : List Nat), commonPrefix a b <+: a
  | [], _ => by simp [commonPrefix]
  | _ :: _, [] => by simp [commonPrefix]
  | x :: as, y :: bs => by
      unfold commonPrefix
      split
      · exact List.prefix_cons_inj x |>.mpr (commonPrefix_prefix_left as bs)
      · exact List.nil_prefix

theorem commonPrefix_prefix_right : ∀ (a b : List Nat), commonPrefix a b <+: b
  | [], _ => by simp [commonPrefix]
  | _ :: _, [] => by simp [commonPrefix]
  | x :: as, y :: bs => by
      unfold commonPrefix
      split
      · rename_i h
        subst h
        exact List.prefix_cons_inj x |>.mpr (commonPrefix_prefix_right as bs)
      · exact List.nil_prefix

theorem commonPrefix_self (a : List Nat) : commonPrefix a a = a := by
  induction a with
  | nil => rfl
  | cons x xs ih =>
      unfold commonPrefix
      simp [ih]

/- ───────── 2. FIFO Ring Buffer (Capacity & Order Preservation) ───────── -/

def push {α : Type} (cap : Nat) (q : List α) (x : α) : Option (List α) :=
  if q.length < cap then some (q ++ [x]) else none

def pop {α : Type} : List α → Option (α × List α)
  | [] => none
  | x :: t => some (x, t)

theorem push_length {α : Type} (cap : Nat) (q : List α) (x : α) (h : q.length < cap) :
    (push cap q x).map List.length = some (q.length + 1) := by
  simp [push, h]

theorem push_full_rejects {α : Type} (cap : Nat) (q : List α) (x : α) (h : ¬ q.length < cap) :
    push cap q x = none := by
  simp [push, h]

theorem pop_is_oldest {α : Type} (x : α) (t : List α) : pop (x :: t) = some (x, t) := rfl

def drain {α : Type} : Nat → List α → List α
  | 0, _ => []
  | n + 1, q => match pop q with
      | none => []
      | some (x, t) => x :: drain n t

theorem drain_all {α : Type} (q : List α) : drain q.length q = q := by
  induction q with
  | nil => rfl
  | cons x t ih => simp [drain, pop, ih]

theorem push_then_drain {α : Type} (cap : Nat) (q : List α) (x : α) (q' : List α)
    (h : push cap q x = some q') : drain q'.length q' = q ++ [x] := by
  have : q' = q ++ [x] := by
    simp [push] at h
    exact h.2.symm
  subst this
  exact drain_all _

end Weft.SubstantiveInvariants
