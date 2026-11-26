-- Pontomedullary Reticular Formation (PMRF) & Reticulospinal Tract (RST) Postural Synergy Invariants
-- Formal verification of Anticipatory Postural Adjustments (APAs), phase-lead temporal ordering,
-- axial stiffness modulation, and Center-of-Mass (CoM) equilibrium preservation.
-- Target #376 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.ReticulospinalPostural

/-- Postural stability state:
    `comPosition`: Center of mass displacement from neutral (scaled as Nat, e.g. mm)
    `bosMargin`: Base of support boundary threshold (scaled as Nat)
    `pmrfDrive`: Reticulospinal axial drive (scaled as Nat, 0-100)
    `distalReachOnset`: Timestamp of distal limb motor initiation (ms)
    `apaOnset`: Timestamp of reticulospinal anticipatory postural activation (ms)
-/
structure PosturalSynergyState where
  comPosition : Nat
  bosMargin : Nat
  pmrfDrive : Nat
  distalReachOnset : Nat
  apaOnset : Nat
  h_bos_pos : bosMargin > 0
  deriving Repr

/-- Compute effective axial stiffness recruited by reticulospinal drive:
    baseStiffness (30) + (pmrfDrive * stiffnessSlope (2)). -/
def computeAxialStiffness (s : PosturalSynergyState) : Nat :=
  30 + s.pmrfDrive * 2

/-- Predicate for Anticipatory Postural Adjustment (APA) temporal phase-lead:
    APA activation occurs strictly before distal reach onset by at least leadMargin (ms). -/
def hasValidAPALead (s : PosturalSynergyState) (leadMargin : Nat) : Bool :=
  s.apaOnset + leadMargin <= s.distalReachOnset

/-- Predicate for Center of Mass equilibrium preservation:
    CoM displacement remains strictly within the Base of Support margin. -/
def isEquilibriumPreserved (s : PosturalSynergyState) : Bool :=
  s.comPosition < s.bosMargin

/-- Theorem 1: Axial stiffness increases monotonically with PMRF reticulospinal drive. -/
theorem axial_stiffness_monotonic (s : PosturalSynergyState) (d1 d2 : Nat)
    (h_le : d1 <= d2) :
    let s1 : PosturalSynergyState := { s with pmrfDrive := d1 }
    let s2 : PosturalSynergyState := { s with pmrfDrive := d2 }
    computeAxialStiffness s1 <= computeAxialStiffness s2 := by
  intro s1 s2
  dsimp [computeAxialStiffness, s1, s2]
  omega

/-- Theorem 2: Valid APA phase-lead invariant:
    If APA onset precedes distal reach by at least leadMargin, any strictly larger
    distal onset delay also satisfies the phase-lead requirement. -/
theorem apa_lead_preserved_under_delay (s : PosturalSynergyState) (leadMargin extraDelay : Nat)
    (h_lead : hasValidAPALead s leadMargin = true) :
    let s_delayed : PosturalSynergyState := { s with distalReachOnset := s.distalReachOnset + extraDelay }
    hasValidAPALead s_delayed leadMargin = true := by
  intro s_delayed
  dsimp [hasValidAPALead, s_delayed]
  dsimp [hasValidAPALead] at h_lead
  have h_ineq : s.apaOnset + leadMargin <= s.distalReachOnset := of_decide_eq_true h_lead
  have h_delayed : s.apaOnset + leadMargin <= s.distalReachOnset + extraDelay := by omega
  exact decide_eq_true h_delayed

/-- Theorem 3: Base of support expansion preserves postural equilibrium:
    If equilibrium is preserved for a base of support margin b1, it is preserved for any b2 >= b1. -/
theorem bos_expansion_preserves_equilibrium (s : PosturalSynergyState) (b1 b2 : Nat)
    (h_b : b1 <= b2)
    (h_pos1 : b1 > 0)
    (h_pos2 : b2 > 0)
    (h_eq1 : isEquilibriumPreserved { s with bosMargin := b1, h_bos_pos := h_pos1 } = true) :
    isEquilibriumPreserved { s with bosMargin := b2, h_bos_pos := h_pos2 } = true := by
  dsimp [isEquilibriumPreserved] at h_eq1
  dsimp [isEquilibriumPreserved]
  have h_comp : s.comPosition < b1 := of_decide_eq_true h_eq1
  have h_comp2 : s.comPosition < b2 := by omega
  exact decide_eq_true h_comp2

/-- Theorem 4: Simultaneous increase in PMRF drive and BoS margin maintains both
    stiffness monotonicity and equilibrium bounds. -/
theorem coordinated_synergy_reinforcement (s : PosturalSynergyState) (d_new b_new : Nat)
    (h_d : s.pmrfDrive <= d_new)
    (h_b : s.bosMargin <= b_new)
    (h_pos : b_new > 0)
    (h_eq : isEquilibriumPreserved s = true) :
    computeAxialStiffness s <= computeAxialStiffness { s with pmrfDrive := d_new } ∧
    isEquilibriumPreserved { s with bosMargin := b_new, h_bos_pos := h_pos } = true := by
  constructor
  · dsimp [computeAxialStiffness]
    omega
  · dsimp [isEquilibriumPreserved] at h_eq
    dsimp [isEquilibriumPreserved]
    have h_c : s.comPosition < s.bosMargin := of_decide_eq_true h_eq
    have h_c2 : s.comPosition < b_new := by omega
    exact decide_eq_true h_c2

end Weft.Formal.ReticulospinalPostural
