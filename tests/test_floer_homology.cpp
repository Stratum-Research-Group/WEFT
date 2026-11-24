#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_floer_homology.h"

int main() {
    std::cout << "[TEST] Running Floer Homology Long-Horizon State Preservation Invariant Test..." << std::endl;

    // Test 1: Action computation on harmonic loop
    const size_t N = 100;
    std::vector<weft::FloerPoint> traj(N);
    float dt = 0.05f;
    for (size_t i = 0; i < N; ++i) {
        float t = i * dt;
        traj[i].q = std::cos(t);
        traj[i].p = -std::sin(t);
    }
    float action = weft::compute_hamiltonian_action(traj.data(), N, dt, 1.0f, 1.0f);
    std::cout << "  Calculated Floer action: " << action << std::endl;
    assert(std::isfinite(action));

    // Test 2: Unbroken Floer cycles counting
    std::vector<float> actions = {1.5f, 0.2f, 3.4f, 0.8f, 2.1f};
    std::vector<int32_t> indices = {1, 0, 2, 1, 3};
    uint32_t cycles = weft::count_unbroken_floer_cycles(actions.data(), indices.data(), actions.size(), 1.0f);
    std::cout << "  Persistent cycles (tau >= 1.0): " << cycles << std::endl;
    assert(cycles == 3);

    // Test 3: In-register trajectory snapping
    const size_t DIM = 32;
    std::vector<float> state(DIM, 2.0f);
    std::vector<float> ref(DIM, 1.0f);
    weft::snap_to_floer_cycle(state.data(), DIM, ref.data(), 0.5f);
    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(state[i] - 1.5f) < 1e-5f);
    }

    std::cout << "[PASS] All Floer Homology invariants verified." << std::endl;
    return 0;
}
