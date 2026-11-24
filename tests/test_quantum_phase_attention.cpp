// test_quantum_phase_attention.cpp
// Validation of unitary state probability norm conservation under phase gates and Hadamard transforms.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_quantum_phase_attention.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_quantum_phase_attention..." << std::endl;

    const int N = 8; // 3-qubit state space
    weft::ComplexState state(N);

    // Initial state |0>: probability 1 at state 0
    state.data[0] = 1.0f; // Real part of state 0
    state.data[1] = 0.0f;

    float init_norm = state.prob_norm();
    std::cout << "[TEST] Initial Probability Norm: " << init_norm << std::endl;
    assert(std::abs(init_norm - 1.0f) < 1e-6f);

    // Apply Hadamard pair between state 0 and state 1
    weft::apply_hadamard_pair(state, 0, 1);
    float post_h_norm = state.prob_norm();
    std::cout << "[TEST] Post-Hadamard Probability Norm: " << post_h_norm << std::endl;
    assert(std::abs(post_h_norm - 1.0f) < 1e-6f);

    // Apply diagonal phase rotations
    std::vector<float> thetas(N);
    for (int j = 0; j < N; ++j) {
        thetas[j] = float(j) * 0.25f;
    }
    weft::apply_diagonal_phase_gate_avx512(state, thetas.data());

    float post_phase_norm = state.prob_norm();
    std::cout << "[TEST] Post-Phase Gate Probability Norm: " << post_phase_norm << std::endl;
    assert(std::abs(post_phase_norm - 1.0f) < 1e-6f);

    std::cout << "[TEST] All Quantum Phase Attention tests passed successfully!" << std::endl;
    return 0;
}
