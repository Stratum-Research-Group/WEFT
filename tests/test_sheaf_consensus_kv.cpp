// test_sheaf_consensus_kv.cpp
// Validation of cellular sheaf Laplacian diffusion and exponential consensus convergence.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_sheaf_consensus_kv.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_sheaf_consensus_kv..." << std::endl;

    const int num_nodes = 4;
    const int stalk_dim = 16;
    weft::SheafGraph graph(num_nodes, stalk_dim);

    // Initial non-consensus states: node u has state = float(u)
    for (int u = 0; u < num_nodes; ++u) {
        float* stalk = graph.stalk(u);
        for (int i = 0; i < stalk_dim; ++i) {
            stalk[i] = float(u);
        }
    }

    float initial_energy = weft::compute_sheaf_energy(graph);
    std::cout << "[TEST] Initial Sheaf Coboundary Energy: " << initial_energy << std::endl;
    assert(initial_energy > 0.0f);

    // Run 50 steps of sheaf Laplacian diffusion with alpha = 0.15
    for (int step = 0; step < 50; ++step) {
        weft::sheaf_diffusion_step_avx512(graph, 0.15f);
    }

    float final_energy = weft::compute_sheaf_energy(graph);
    std::cout << "[TEST] Final Sheaf Coboundary Energy (after 50 steps): " << final_energy << std::endl;

    assert(final_energy < initial_energy * 0.01f); // At least 99% reduction in consensus error
    std::cout << "[TEST] All Sheaf Cohomology KV Consensus tests passed successfully!" << std::endl;
    return 0;
}
