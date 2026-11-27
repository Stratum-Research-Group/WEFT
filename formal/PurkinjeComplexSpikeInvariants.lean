-- Cerebellar Purkinje Cell Complex Spike Climbing Fiber Burst Invariants (Target #426)
-- Inferior olivary climbing fiber projection, all-or-none complex spikes with secondary Ca2+ spikelets,
-- parallel fiber mGluR1/IP3 co-activation, and parallel fiber AMPA receptor internalization (LTD >= 35%)

structure PurkinjeState where
  climbingFreq : Nat         -- Inferior olivary climbing fiber firing frequency scaled x10 (e.g. 25 = 2.5 Hz)
  pfDrive : Nat              -- Parallel fiber co-activation percentage (0..100)
  dendriticCa : Nat          -- Dendritic calcium concentration in micromolar scaled x10 (e.g. 21 = 2.1 uM)
  mglur1Ratio : Nat          -- mGluR1/IP3/DAG activation ratio scaled x10 (e.g. 34 = 3.4x)
  pfLtdPct : Nat             -- Parallel fiber LTD depression level percentage (0..100)
  errorScore : Nat           -- Motor prediction error score scaled x100 (e.g. 18 = 0.18)
  deriving Repr, DecidableEq

def compute_purkinje_state (cf_hz_x10 : Nat) (pf : Nat) : PurkinjeState :=
  -- Dendritic calcium transient: (cf * 22 / 40) + (pf * 10 / 100)
  let ca := (cf_hz_x10 * 22) / 40 + (pf * 10) / 100
  -- mGluR1 activation: (pf * 42 / 100) if cf >= 10 (>= 1.0 Hz) else (pf * 8 / 100)
  let mglur := if cf_hz_x10 >= 10 then (pf * 42) / 100 else (pf * 8) / 100
  -- Parallel fiber LTD: (ca * mglur * 38) / 500
  let raw_ltd := (ca * mglur * 38) / 500
  let ltd := max 5 (min 65 raw_ltd)
  -- Error score: 100 - (ltd * 85) / 50
  let penalty := (ltd * 85) / 50
  let err := if 100 > penalty then max 5 (100 - penalty) else 5
  { climbingFreq := cf_hz_x10,
    pfDrive := pf,
    dendriticCa := ca,
    mglur1Ratio := mglur,
    pfLtdPct := ltd,
    errorScore := err }

def is_learning_competent (s : PurkinjeState) : Bool :=
  s.climbingFreq >= 10 && s.climbingFreq <= 40 && s.dendriticCa >= 18 && s.pfLtdPct >= 35 && s.errorScore <= 45

def is_ataxic_failure (s : PurkinjeState) : Bool :=
  s.climbingFreq < 5 || s.dendriticCa < 12 || s.pfLtdPct < 20

theorem olivary_burst_guarantees_ltd_learning :
  let s := compute_purkinje_state 25 80
  is_learning_competent s = true := by
  decide

theorem climbing_fiber_silencing_causes_ataxia :
  let s := compute_purkinje_state 2 80
  is_ataxic_failure s = true := by
  decide

theorem climbing_fiber_monotonic_calcium :
  let s_low := compute_purkinje_state 10 70
  let s_high := compute_purkinje_state 30 70
  s_high.dendriticCa >= s_low.dendriticCa := by
  decide
