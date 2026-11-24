#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_motivic_lfunctions.h"

int main() {
    std::cout << "[TEST] Running Motivic L-Functions & Euler Systems Invariant Test..." << std::endl;

    const size_t DIM = 16;
    std::vector<float> curr_w(DIM, 1.0f);
    std::vector<float> grad(DIM, 0.5f);
    std::vector<float> updated_w(DIM, 0.0f);

    float base_lr = 0.1f;
    uint32_t rank = 3; // rank 3 -> damping = 1/(1+3) = 0.25 -> lr_eff = 0.025

    weft::MotivicDescentProfile prof = weft::compute_motivic_euler_descent_step(
        grad.data(), updated_w.data(), curr_w.data(), DIM, base_lr, rank
    );

    std::cout << "  Base learning rate: " << base_lr << std::endl;
    std::cout << "  Analytic rank damping: " << prof.analytic_rank_damping << " (expected: 0.25)" << std::endl;
    std::cout << "  Calibrated step size: " << prof.calibrated_step_size << " (expected: 0.025)" << std::endl;
    std::cout << "  Gradient norm: " << prof.gradient_norm << std::endl;

    assert(std::abs(prof.analytic_rank_damping - 0.25f) < 1e-5f);
    assert(std::abs(prof.calibrated_step_size - 0.025f) < 1e-5f);
    // Updated weight: 1.0 - 0.025 * 0.5 = 1.0 - 0.0125 = 0.9875
    assert(std::abs(updated_w[0] - 0.9875f) < 1e-5f);

    std::cout << "[PASS] All Motivic L-Functions & Euler Systems invariants verified." << std::endl;
    return 0;
}
