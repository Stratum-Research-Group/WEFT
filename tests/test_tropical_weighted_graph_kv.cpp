// test_tropical_weighted_graph_kv.cpp
// Milestone Paper 250: Verification of AVX-512 Tropical Weighted Graph KV Eviction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_weighted_graph_kv.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_cluster_eviction_kernel() {
    const size_t N = 32;
    std::vector<float> bridge_distances(N, 0.0f);
    // First 16 clusters are nearby (dist = 5.0f)
    for (size_t i = 0; i < 16; ++i) {
        bridge_distances[i] = 5.0f;
    }
    // Last 16 clusters are distant document subgraphs (dist = 50.0f)
    for (size_t i = 16; i < 32; ++i) {
        bridge_distances[i] = 50.0f;
    }

    std::vector<uint8_t> evict_flags(N, 0);
    float tau_bridge = 20.0f;

    weft::avx512_evaluate_cluster_eviction(bridge_distances.data(), evict_flags.data(), N, tau_bridge);

    // Invariant: nearby clusters kept (flag = 0), distant evicted (flag = 1)
    for (size_t i = 0; i < 16; ++i) {
        assert(evict_flags[i] == 0);
    }
    for (size_t i = 16; i < 32; ++i) {
        assert(evict_flags[i] == 1);
    }
}

int main() {
    std::cout << "[Paper 250] Running Tropical Weighted Graph KV Eviction Unit Tests...\n";
    test_cluster_eviction_kernel();
    std::cout << "[Paper 250] All unit tests PASSED successfully.\n";
    return 0;
}
