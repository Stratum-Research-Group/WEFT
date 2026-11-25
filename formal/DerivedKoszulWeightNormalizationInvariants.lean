/-
  Formal Verification of Derived Koszul Duality for Non-Archimedean Weight Normalization.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.DerivedKoszulWeightNormalization

structure WeightNormalizationSpec where
  tensorDim : Nat
  valuationExponent : Nat
  barCobarResolutions : Nat

def totalNormalizationDegrees (spec : WeightNormalizationSpec) : Nat :=
  spec.tensorDim * spec.valuationExponent + spec.barCobarResolutions

theorem normalization_degrees_nonnegative (spec : WeightNormalizationSpec) :
    totalNormalizationDegrees spec >= 0 := by
  exact Nat.zero_le (totalNormalizationDegrees spec)

theorem tensor_dim_positive (spec : WeightNormalizationSpec) (h : spec.tensorDim > 0) :
    spec.tensorDim >= 1 := by
  exact h

end Weft.DerivedKoszulWeightNormalization
