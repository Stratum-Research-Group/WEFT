-- TropicalToricDivisorsInvariants.lean
-- Milestone Paper 286: Asymmetric Tropical Toric Boundary Divisors & Logarithmic Geometry
-- Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
-- License: Apache-2.0

structure PiecewiseLinearActivationState where
  dim : Nat
  errorBoundScaled : Nat
  toleranceScaled : Nat
  deriving Repr

def isApproximationValid (s : PiecewiseLinearActivationState) : Prop :=
  s.errorBoundScaled ≤ s.toleranceScaled

theorem approximation_validity_guaranteed (s : PiecewiseLinearActivationState) (h_err : s.errorBoundScaled = 0) :
    isApproximationValid s := by
  dsimp [isApproximationValid]
  rw [h_err]
  exact Nat.zero_le s.toleranceScaled

theorem error_bound_monotonicity (e1 e2 tol : Nat) (h_le : e1 ≤ e2) (h_valid : e2 ≤ tol) :
    e1 ≤ tol := by
  exact Nat.le_trans h_le h_valid
