/-
  Formal Verification of Perfectoid Cluster MoE Routing Invariants.
  Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0
-/

namespace Weft.PerfectoidClusterMoERouting

structure ClusterMoESpec where
  numClusters : Nat
  expertsPerCluster : Nat
  activeClusters : Nat

def totalClusterRoutingDim (spec : ClusterMoESpec) : Nat :=
  spec.numClusters * spec.expertsPerCluster + spec.activeClusters

theorem cluster_routing_nonnegative (spec : ClusterMoESpec) :
    totalClusterRoutingDim spec >= 0 := by
  exact Nat.zero_le (totalClusterRoutingDim spec)

theorem cluster_routing_positive (spec : ClusterMoESpec) (hc : spec.numClusters > 0) (he : spec.expertsPerCluster > 0) :
    totalClusterRoutingDim spec >= 1 := by
  dsimp [totalClusterRoutingDim]
  have h1 : spec.numClusters * spec.expertsPerCluster >= 1 := Nat.succ_le_of_lt (Nat.mul_pos hc he)
  exact Nat.le_add_right_of_le h1

end Weft.PerfectoidClusterMoERouting
