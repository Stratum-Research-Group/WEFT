-- TropicalBerkovichKVCompactionInvariants.lean
-- Milestone Paper 259: Asymmetric Tropical Berkovich Spectra & Non-Archimedean Amoebas
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure KVPage where
  pageId : Nat
  occupancy : Nat
  maxCapacity : Nat
  h_cap : occupancy <= maxCapacity
  deriving Repr

def ultrametricMax (v1 v2 : Nat) : Nat :=
  if v1 >= v2 then v1 else v2

theorem ultrametric_max_ge_left (v1 v2 : Nat) :
    ultrametricMax v1 v2 >= v1 := by
  dsimp [ultrametricMax]
  split <;> omega

theorem ultrametric_max_ge_right (v1 v2 : Nat) :
    ultrametricMax v1 v2 >= v2 := by
  dsimp [ultrametricMax]
  split <;> omega

def isPageCompactable (p : KVPage) (threshold : Nat) : Bool :=
  p.occupancy <= threshold

def mergePages (p1 p2 : KVPage) : Nat :=
  p1.occupancy + p2.occupancy

theorem merge_pages_conservation (p1 p2 : KVPage) :
    mergePages p1 p2 = p1.occupancy + p2.occupancy := by
  dsimp [mergePages]

theorem merged_occupancy_nonneg (p1 p2 : KVPage) :
    mergePages p1 p2 >= 0 := by
  exact Nat.zero_le _
