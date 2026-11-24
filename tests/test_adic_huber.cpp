#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_adic_huber.h"

int main() {
    std::cout << "[TEST] Running Adic Spaces & Huber Rings Invariant Test..." << std::endl;

    const size_t NUM_LAYERS = 64;
    // Layer residuals: each residual has magnitude 0.1
    std::vector<float> residuals(NUM_LAYERS, 0.1f);

    weft::AdicResidualProfile prof = weft::track_adic_residual_stream(
        residuals.data(), NUM_LAYERS
    );

    std::cout << "  Max Ultrametric Valuation: " << prof.max_valuation << " (expected 0.1)" << std::endl;
    std::cout << "  Cumulative Archimedean L2: " << prof.cumulative_archimedean_l2 << " (expected 0.8)" << std::endl;
    std::cout << "  Ultrametric Bound Ratio: " << prof.ultrametric_bound_ratio << std::endl;

    assert(std::abs(prof.max_valuation - 0.1f) < 1e-5f);
    assert(std::abs(prof.cumulative_archimedean_l2 - 0.8f) < 1e-4f);
    // Ultrametric valuation strictly bounds individual layer divergence
    assert(prof.max_valuation <= prof.cumulative_archimedean_l2);

    std::cout << "[PASS] All Adic Spaces & Huber Rings invariants verified." << std::endl;
    return 0;
}
