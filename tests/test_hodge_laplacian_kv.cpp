// test_hodge_laplacian_kv.cpp
// Validation of discrete Hodge Laplacian projection and circulation removal.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_hodge_laplacian_kv.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_hodge_laplacian_kv..." << std::endl;

    // Initial flow with strong circulation: omega = [2.0, 2.0, 2.0] -> curl = 6.0
    weft::TriangularMemoryComplex comp(2.0f, 2.0f, 2.0f);

    float init_curl = comp.curl();
    std::cout << "[TEST] Initial Circulation Curl: " << init_curl << " (Expected 6.0)" << std::endl;
    assert(std::abs(init_curl - 6.0f) < 1e-6f);

    float init_energy = weft::compute_hodge_energy(comp);
    std::cout << "[TEST] Initial Hodge Energy: " << init_energy << std::endl;

    // Remove circulation via orthogonal Hodge projection
    weft::remove_circulation_avx512(comp);

    float post_curl = comp.curl();
    std::cout << "[TEST] Post-Projection Circulation Curl: " << post_curl << " (Expected 0.0)" << std::endl;
    assert(std::abs(post_curl) < 1e-6f);

    // Flow is now purely circulation-free and balanced: omega = [0.0, 0.0, 0.0]
    for (int i = 0; i < 3; ++i) {
        assert(std::abs(comp.omega[i]) < 1e-6f);
    }
    std::cout << "[TEST] Circulation Component Fully Extracted." << std::endl;

    std::cout << "[TEST] All Hodge Laplacian KV Flow tests passed successfully!" << std::endl;
    return 0;
}
