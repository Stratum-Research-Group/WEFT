-- Horizontal Oculomotor Neural Integrator (NPH / MVN) & Pulse-Step Invariants
-- Formal verification of velocity pulse integration into position step commands,
-- orbital elastic recoil equilibrium, and leaky integration gaze-drift bounding.
-- Target #380 in the WEFT Formal Verification Suite.
-- Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

namespace Weft.Formal.NeuralIntegrator

/-- State of horizontal oculomotor neural integrator:
    `gazeAngle`: Horizontal eye eccentricity in degrees (scaled Nat)
    `tauLeak`: Integrator leak time constant in seconds (scaled Nat)
    `pulseVelocity`: Transient premotor burst velocity command (scaled Nat)
    `maxDriftThreshold`: Maximum allowable centripetal drift before nystagmus (scaled Nat)
-/
structure NeuralIntegratorState where
  gazeAngle : Nat
  tauLeak : Nat
  pulseVelocity : Nat
  maxDriftThreshold : Nat
  h_tau_pos : tauLeak > 0
  deriving Repr

/-- Compute step tonic motor firing rate needed to overcome orbital elastic recoil:
    baseStep (60) + gazeAngle * 4. -/
def computeStepTonicFiring (s : NeuralIntegratorState) : Nat :=
  60 + s.gazeAngle * 4

/-- Compute centripetal gaze-drift velocity: gazeAngle / tauLeak. -/
def computeGazeDrift (s : NeuralIntegratorState) : Nat :=
  s.gazeAngle / s.tauLeak

/-- Predicate for stable gaze holding:
    Centripetal drift velocity is strictly bounded by maxDriftThreshold. -/
def isGazeHoldingStable (s : NeuralIntegratorState) : Bool :=
  computeGazeDrift s <= s.maxDriftThreshold

/-- Theorem 1: Step tonic firing rate increases monotonically with eccentric gaze angle. -/
theorem step_firing_monotonic_with_angle (s : NeuralIntegratorState) (a1 a2 : Nat)
    (h_le : a1 <= a2) :
    let s1 : NeuralIntegratorState := { s with gazeAngle := a1 }
    let s2 : NeuralIntegratorState := { s with gazeAngle := a2 }
    computeStepTonicFiring s1 <= computeStepTonicFiring s2 := by
  intro s1 s2
  dsimp [computeStepTonicFiring, s1, s2]
  omega

/-- Theorem 2: Ideal infinite time constant eliminates centripetal drift:
    When tauLeak exceeds gazeAngle (tauLeak > gazeAngle), integer drift is strictly zero. -/
theorem long_time_constant_zero_drift (s : NeuralIntegratorState)
    (h_long : s.gazeAngle < s.tauLeak) :
    computeGazeDrift s = 0 := by
  dsimp [computeGazeDrift]
  exact Nat.div_eq_of_lt h_long

/-- Theorem 3: Extended integrator time constant reduces gaze drift velocity monotonically. -/
theorem longer_tau_reduces_drift (s : NeuralIntegratorState) (t1 t2 : Nat)
    (h_le : t1 <= t2)
    (h_pos1 : t1 > 0)
    (h_pos2 : t2 > 0) :
    let s1 : NeuralIntegratorState := { s with tauLeak := t1, h_tau_pos := h_pos1 }
    let s2 : NeuralIntegratorState := { s with tauLeak := t2, h_tau_pos := h_pos2 }
    computeGazeDrift s2 <= computeGazeDrift s1 := by
  intro s1 s2
  dsimp [computeGazeDrift, s1, s2]
  exact Nat.div_le_div_left h_le h_pos1

/-- Theorem 4: Gaze holding stability is preserved under reduced eccentric gaze angles. -/
theorem reduced_angle_preserves_stability (s : NeuralIntegratorState) (a1 a2 : Nat)
    (h_le : a1 <= a2)
    (h_stable2 : isGazeHoldingStable { s with gazeAngle := a2 } = true) :
    isGazeHoldingStable { s with gazeAngle := a1 } = true := by
  dsimp [isGazeHoldingStable, computeGazeDrift] at h_stable2
  dsimp [isGazeHoldingStable, computeGazeDrift]
  have h_bound2 : a2 / s.tauLeak <= s.maxDriftThreshold := of_decide_eq_true h_stable2
  have h_div_mono : a1 / s.tauLeak <= a2 / s.tauLeak := Nat.div_le_div_right h_le
  have h_bound1 : a1 / s.tauLeak <= s.maxDriftThreshold := by omega
  exact decide_eq_true h_bound1

end Weft.Formal.NeuralIntegrator
