-- Striatal Medium Spiny Neuron (MSN) D1/D2 Dopaminergic Plasticity Invariants (Target #424)
-- Direct (D1/Golf/PKA/LTP) vs. Indirect (D2/Gi/Calcineurin/LTD) pathway gain balance,
-- motor initiation gating, parkinsonian akinesia, and dyskinesia thresholds

structure StriatalPlasticityState where
  daConcNm : Nat           -- Striatal extracellular dopamine in nM (0..2000 nM)
  glutDrive : Nat          -- Corticostriatal glutamate drive percentage (0..100)
  pkaRatio : Nat           -- D1 PKA phosphorylation ratio scaled x10 (e.g. 36 = 3.6x)
  calcineurinRatio : Nat   -- D2 Calcineurin activation ratio scaled x10 (e.g. 16 = 1.6x)
  d1Gain : Nat             -- Direct pathway gain scaled x100 (e.g. 202 = 2.02)
  d2Gain : Nat             -- Indirect pathway gain scaled x100 (e.g. 54 = 0.54)
  gainRatio : Nat          -- D1/D2 pathway gain ratio scaled x100 (e.g. 374 = 3.74)
  deriving Repr, DecidableEq

def compute_striatal_state (da_nm : Nat) (glut : Nat) : StriatalPlasticityState :=
  -- D1 PKA ratio scaled x10: 5 + (da * 45) / (da + 600)
  let pka := 5 + (da_nm * 45) / (da_nm + 600)
  -- D2 Calcineurin ratio scaled x10: 45 - (da * 35) / (da + 150)
  let cn_drop := (da_nm * 35) / (da_nm + 150)
  let cn := if 45 > cn_drop then 45 - cn_drop else 10
  -- Direct (D1) and Indirect (D2) pathway gains scaled x100
  let d1 := (glut * pka * 75) / 1000
  let d2 := (glut * cn * 45) / 1000
  -- Gain ratio scaled x100: (d1 * 100) / max 1 d2
  let ratio := (d1 * 100) / (max 1 d2)
  { daConcNm := da_nm,
    glutDrive := glut,
    pkaRatio := pka,
    calcineurinRatio := cn,
    d1Gain := d1,
    d2Gain := d2,
    gainRatio := ratio }

def is_normal_motor_initiation (s : StriatalPlasticityState) : Bool :=
  s.daConcNm >= 600 && s.gainRatio >= 150 && s.gainRatio <= 450

def is_parkinsonian_akinesia (s : StriatalPlasticityState) : Bool :=
  s.daConcNm < 250 || s.gainRatio < 100

def is_dyskinesia_chorea (s : StriatalPlasticityState) : Bool :=
  s.daConcNm > 1500 || s.gainRatio > 450

theorem normal_dopamine_burst_guarantees_initiation :
  let s := compute_striatal_state 850 75
  is_normal_motor_initiation s = true := by
  decide

theorem dopamine_depletion_causes_akinesia :
  let s := compute_striatal_state 80 75
  is_parkinsonian_akinesia s = true := by
  decide

theorem dopamine_monotonic_d1_gain :
  let s_low := compute_striatal_state 200 70
  let s_high := compute_striatal_state 900 70
  s_high.d1Gain >= s_low.d1Gain := by
  decide
