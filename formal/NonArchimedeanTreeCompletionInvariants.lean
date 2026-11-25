/-
  Formal Verification of Non-Archimedean Tree Completion for Speculative Streaming.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.NonArchimedeanTreeCompletion

structure TreeCompletionSpec where
  treeLeaves : Nat
  completionDepth : Nat
  streamChannels : Nat

def totalCompletionDegrees (spec : TreeCompletionSpec) : Nat :=
  spec.treeLeaves * spec.completionDepth + spec.streamChannels

theorem tree_completion_nonnegative (spec : TreeCompletionSpec) :
    totalCompletionDegrees spec >= 0 := by
  exact Nat.zero_le (totalCompletionDegrees spec)

theorem tree_completion_positive (spec : TreeCompletionSpec) (hl : spec.treeLeaves > 0) (hd : spec.completionDepth > 0) :
    totalCompletionDegrees spec >= 1 := by
  dsimp [totalCompletionDegrees]
  have h1 : spec.treeLeaves * spec.completionDepth >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hl hd)
  exact Nat.le_add_right_of_le h1

end Weft.NonArchimedeanTreeCompletion
