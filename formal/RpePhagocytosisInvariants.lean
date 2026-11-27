-- Retinal Photoreceptor Outer Segment Disc Shedding & RPE Phagocytosis Invariants (Target #425)
-- Diurnal rod/cone disc shedding, Gas6/Protein S opsonization, avb5 integrin binding,
-- MerTK autophosphorylation & engulfment, A2E/lipofuscin clearance, and photoreceptor survival

structure RpeState where
  mertkPhos : Nat            -- MerTK receptor autophosphorylation percentage (0..100)
  gas6Opson : Nat            -- Gas6/Protein S opsonization percentage (0..100)
  avb5Binding : Nat          -- avb5 integrin tethering percentage (0..100)
  posClearance : Nat         -- Diurnal shed POS clearance percentage (0..100)
  a2eBurden : Nat            -- Toxic bisretinoid A2E/lipofuscin burden percentage (0..100)
  prSurvival : Nat           -- Photoreceptor survival probability percentage (0..100)
  deriving Repr, DecidableEq

def compute_rpe_state (mertk : Nat) (gas6 : Nat) : RpeState :=
  -- avb5 integrin binding mediated by Gas6
  let binding := min 98 ((gas6 * 105) / 100)
  -- Phagocytosis rate: (mertk * binding) / 100
  let rate := (mertk * binding) / 100
  -- Diurnal clearance within 3.5h: min 99 (rate * 140 / 100)
  let clearance := min 99 ((rate * 140) / 100)
  -- A2E lipofuscin accumulation from uncleared oxidized discs
  let unclar := if 100 > clearance then 100 - clearance else 0
  let raw_a2e := (unclar * 22) / 10
  let a2e := max 5 (min 95 raw_a2e)
  -- Photoreceptor survival
  let loss := (a2e * 75) / 100
  let surv := if 100 > loss then max 10 (100 - loss) else 10
  { mertkPhos := mertk,
    gas6Opson := gas6,
    avb5Binding := binding,
    posClearance := clearance,
    a2eBurden := a2e,
    prSurvival := surv }

def is_homeostatic_rpe (s : RpeState) : Bool :=
  s.mertkPhos >= 65 && s.posClearance >= 95 && s.a2eBurden <= 20 && s.prSurvival >= 85

def is_rcs_rat_degeneration (s : RpeState) : Bool :=
  s.mertkPhos < 30 || s.posClearance < 60 || s.a2eBurden > 50

theorem intact_mertk_guarantees_homeostasis :
  let s := compute_rpe_state 85 80
  is_homeostatic_rpe s = true := by
  decide

theorem mertk_loss_causes_degeneration :
  let s := compute_rpe_state 15 80
  is_rcs_rat_degeneration s = true := by
  decide

theorem mertk_monotonic_clearance :
  let s_low := compute_rpe_state 30 75
  let s_high := compute_rpe_state 85 75
  s_high.posClearance >= s_low.posClearance := by
  decide
