// test_contact_kv.cpp
// Validation of contact Hamiltonian dynamics and monotonic energy dissipation.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_contact_kv.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_contact_kv..." << std::endl;

    const int dim = 64;
    weft::ContactState state(dim);

    // Initial non-zero position and momentum
    for (int i = 0; i < dim; ++i) {
        state.q[i] = 1.0f;
        state.p[i] = 1.0f;
    }

    float initial_energy = state.total_energy();
    std::cout << "[TEST] Initial Contact Energy E_0: " << initial_energy << std::endl;
    assert(initial_energy > 0.0f);

    // Advance 100 contact leapfrog steps with damping gamma = 0.05
    const float dt = 0.01f;
    const float gamma = 0.05f;

    float prev_energy = initial_energy;
    for (int step = 0; step < 100; ++step) {
        weft::contact_leapfrog_step_avx512(state, dt, gamma);
        float current_energy = state.total_energy();
        // Energy must be monotonically non-increasing (within machine epsilon)
        assert(current_energy <= prev_energy + 1e-4f);
        prev_energy = current_energy;
    }

    float final_energy = state.total_energy();
    std::cout << "[TEST] Final Contact Energy (after 100 damped steps): " << final_energy << std::endl;
    assert(final_energy < initial_energy);
    std::cout << "[TEST] Dissipation rate strictly bounded and non-positive." << std::endl;

    std::cout << "[TEST] All Contact Geometry KV tests passed successfully!" << std::endl;
    return 0;
}
