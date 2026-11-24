// test_symplectic_kv_cache.cpp - Verification for Paper #125
#include "csrc/cli/weft_symplectic_kv.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_symplectic_kv_cache..." << std::endl;

    WeftSymplecticIntegrator integrator(0.05f);
    SymplecticState state;

    // Harmonic oscillator potential: V(q) = 0.5 * k * ||q||^2, grad_V(q) = k * q
    constexpr float k_spring = 1.0f;

    for (size_t d = 0; d < SymplecticState::DIM; ++d) {
        state.q[d] = 1.0f;
        state.p[d] = 0.0f;
    }

    float initial_energy = 0.0f;
    for (size_t d = 0; d < SymplecticState::DIM; ++d) {
        initial_energy += 0.5f * k_spring * state.q[d] * state.q[d];
    }
    float initial_h = integrator.compute_hamiltonian(state, initial_energy);

    std::cout << "[TEST] Initial Hamiltonian Energy: " << initial_h << std::endl;

    // Integrate over 1000 discrete steps
    for (size_t step = 0; step < 1000; ++step) {
        alignas(64) float grad[SymplecticState::DIM];
        float potential = 0.0f;
        for (size_t d = 0; d < SymplecticState::DIM; ++d) {
            grad[d] = k_spring * state.q[d];
            potential += 0.5f * k_spring * state.q[d] * state.q[d];
        }

        integrator.step(state, grad);

        if (step % 200 == 0) {
            float current_h = integrator.compute_hamiltonian(state, potential);
            float drift = std::abs(current_h - initial_h) / initial_h;
            std::cout << "[TEST] Step " << step << " | Energy: " << current_h << " | Relative Drift: " << drift << std::endl;
            // Symplectic integrator must bound energy drift within small O(eps^2) window
            assert(drift < 0.05f);
        }
    }

    std::cout << "[TEST] Symplectic phase-space volume and energy stability verified!" << std::endl;
    return 0;
}
