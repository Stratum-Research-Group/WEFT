// test_hamiltonian_ssm_routing.cpp
// Numerical energy conservation test for AVX-512 symplectic leapfrog routing.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_hamiltonian_ssm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_hamiltonian_ssm_routing..." << std::endl;

    const int dim = 128;
    weft::HamiltonianState state(dim);

    // Initial non-zero position and momentum
    for (int i = 0; i < dim; ++i) {
        state.q[i] = std::sin(float(i));
        state.p[i] = std::cos(float(i));
    }

    const float initial_energy = weft::compute_hamiltonian_energy(state);
    std::cout << "[TEST] Initial Hamiltonian Energy: " << initial_energy << std::endl;

    // Advance 500 discrete leapfrog steps with dt = 0.01
    const float dt = 0.01f;
    for (int step = 0; step < 500; ++step) {
        weft::leapfrog_step_avx512(state, dt);
    }

    const float final_energy = weft::compute_hamiltonian_energy(state);
    std::cout << "[TEST] Final Hamiltonian Energy (after 500 steps): " << final_energy << std::endl;

    const float relative_drift = std::abs(final_energy - initial_energy) / initial_energy;
    std::cout << "[TEST] Relative Energy Drift: " << relative_drift << " (Bound: < 1e-3)" << std::endl;

    assert(relative_drift < 1e-3f);
    std::cout << "[TEST] All Symplectic Hamiltonian SSM tests passed successfully!" << std::endl;
    return 0;
}
