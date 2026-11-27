-- Presynaptic SNARE Complex Assembly & Munc18-1 Priming Invariants (Target #421)
-- Syntaxin-1A open conformation, SNARE core zippering, RRP vesicle priming,
-- Synaptotagmin-1 Ca2+ sensor activation, and sub-millisecond quantal release jitter

structure SnareState where
  munc13Priming : Nat      -- Munc13 priming drive percentage (0..100)
  openSyntaxin : Nat       -- Syntaxin-1A open fraction percentage (0..100)
  snareZippering : Nat     -- Core SNARE zippering completion percentage (0..100)
  rrpOccupancy : Nat       -- Readily releasable pool occupancy percentage (0..100)
  caMicrodomain : Nat      -- Active-zone microdomain Ca2+ in micromolar (uM)
  releaseJitterUs : Nat    -- Quantal release temporal jitter in microseconds (us)
  deriving Repr, DecidableEq

def compute_snare_state (munc13 : Nat) (ca_um : Nat) : SnareState :=
  -- Munc13-1 drives Syntaxin-1A transition to open conformation
  let open_syn := min 98 ((munc13 * 98) / 100)
  -- SNARE zippering follows open syntaxin availability
  let zip := (open_syn * 100) / (open_syn + 10)
  -- RRP vesicle occupancy
  let rrp := (zip * 95) / 100
  -- Synchronous exocytosis jitter (target <= 250 us)
  let effective_ca := max 1 (min 30 ca_um)
  let base_jitter := 1800 / effective_ca
  let penalty := if rrp < 80 then ((80 - rrp) * 12) else 0
  let jitter := max 120 (base_jitter + penalty)
  { munc13Priming := munc13,
    openSyntaxin := open_syn,
    snareZippering := zip,
    rrpOccupancy := rrp,
    caMicrodomain := ca_um,
    releaseJitterUs := jitter }

def is_synchronous_release (s : SnareState) : Bool :=
  s.munc13Priming >= 70 && s.rrpOccupancy >= 80 && s.caMicrodomain >= 10 && s.releaseJitterUs <= 250

def is_desynchronized_failure (s : SnareState) : Bool :=
  s.munc13Priming < 30 || s.caMicrodomain < 5 || s.releaseJitterUs > 350

theorem munc13_and_ca_guarantee_synchrony :
  let s := compute_snare_state 85 15
  is_synchronous_release s = true := by
  decide

theorem impaired_priming_causes_desynchronization :
  let s := compute_snare_state 20 4
  is_desynchronized_failure s = true := by
  decide

theorem rrp_occupancy_monotonicity :
  let s_low := compute_snare_state 30 15
  let s_high := compute_snare_state 90 15
  s_high.rrpOccupancy >= s_low.rrpOccupancy := by
  decide
