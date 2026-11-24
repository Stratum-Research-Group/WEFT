-- TropicalConvexityKVInvariants.lean
-- Milestone Paper 268: Asymmetric Tropical Convexity & Polytopal Core Allocations
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure TenantAddress where
  tenantId : Nat
  pageBase : Nat
  pageSize : Nat
  deriving Repr

def tropicalMaxComb (x y a b : Nat) : Nat :=
  let branchA := x + a
  let branchB := y + b
  if branchA >= branchB then branchA else branchB

theorem tropical_comb_idempotent (x a : Nat) :
    tropicalMaxComb x x a a = x + a := by
  dsimp [tropicalMaxComb]
  split <;> omega

theorem tropical_comb_nonneg (x y a b : Nat) :
    tropicalMaxComb x y a b >= 0 := by
  exact Nat.zero_le _

def areTenantsDisjoint (t1 t2 : TenantAddress) : Prop :=
  t1.pageBase + t1.pageSize <= t2.pageBase ∨ t2.pageBase + t2.pageSize <= t1.pageBase

theorem disjoint_tenants_no_overlap (t1 t2 : TenantAddress) (addr : Nat)
    (h_disj : areTenantsDisjoint t1 t2)
    (h_t1 : addr >= t1.pageBase ∧ addr < t1.pageBase + t1.pageSize) :
    ¬ (addr >= t2.pageBase ∧ addr < t2.pageBase + t2.pageSize) := by
  dsimp [areTenantsDisjoint] at h_disj
  cases h_disj with
  | inl h1 =>
    intro h_t2
    have h_lt := h_t1.right
    have h_ge := h_t2.left
    omega
  | inr h2 =>
    intro h_t2
    have h_lt := h_t2.right
    have h_ge := h_t1.left
    omega
