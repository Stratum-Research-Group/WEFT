-- Target #427: Hypothalamic AgRP/POMC Metabolic Energy Homeostasis Invariants
-- Formal verification of arcuate nucleus AgRP vs POMC competitive antagonism,
-- MC4R melanocortin signaling, leptin/ghrelin oppositional gating, and autonomic energy expenditure.
-- Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

structure MetabolicState where
  ghrelinDrive : Nat        -- [0, 100] scale
  leptinDrive : Nat         -- [0, 100] scale
  agrpFiring : Nat          -- AgRP firing rate (Hz x 10)
  pomcFiring : Nat          -- POMC firing rate (Hz x 10)
  mc4rTone : Nat            -- MC4R signaling level (%)
  energyExpenditure : Nat   -- Metabolic rate (%)
  homeostaticError : Nat    -- Error (x 100)
deriving Repr, DecidableEq

def compute_metabolic_state (ghrelin leptin : Nat) : MetabolicState :=
  let agrp_raw := ((ghrelin * 38) / 10) + 50
  let agrp := if agrp_raw >= ((leptin * 15) / 10) then agrp_raw - ((leptin * 15) / 10) else 10
  let pomc_raw := ((leptin * 28) / 10) + 30
  let pomc := if pomc_raw >= ((agrp * 25) / 100) then pomc_raw - ((agrp * 25) / 100) else 5
  let mc4r_raw := (pomc * 42) / 10 + 300
  let mc4r_sub := (agrp * 12) / 10
  let mc4r := if mc4r_raw >= mc4r_sub then mc4r_raw - mc4r_sub else 50
  let mc4r_bounded := if mc4r > 1000 then 1000 else if mc4r < 50 then 50 else mc4r
  let ee := (mc4r_bounded * 85) / 100 + 50
  let diff := if (agrp * 10) >= (pomc * 25) then (agrp * 10) - (pomc * 25) else (pomc * 25) - (agrp * 10)
  let err := diff / 25
  {
    ghrelinDrive := ghrelin,
    leptinDrive := leptin,
    agrpFiring := agrp,
    pomcFiring := pomc,
    mc4rTone := mc4r_bounded / 10,
    energyExpenditure := ee / 10,
    homeostaticError := err
  }

def is_satiated (s : MetabolicState) : Bool :=
  s.pomcFiring >= 100 && s.mc4rTone >= 40 && s.energyExpenditure >= 45

def is_hunger_stable (s : MetabolicState) : Bool :=
  s.agrpFiring <= 450 && s.homeostaticError <= 150

theorem satiety_regime_verified :
    let s := compute_metabolic_state 15 85
    is_satiated s = true := by
  decide

theorem hunger_stability_verified :
    let s := compute_metabolic_state 75 25
    is_hunger_stable s = true := by
  decide
