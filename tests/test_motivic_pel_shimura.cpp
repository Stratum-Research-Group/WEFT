#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "csrc/cli/weft_motivic_pel_shimura.h"

int main() {
    std::cout << "[TEST] Running test_motivic_pel_shimura (Paper 311)..." << std::endl;

    const size_t half_dim = 32;
    std::vector<float> x(2 * half_dim, 0.0f);
    std::vector<float> y(2 * half_dim, 0.0f);

    for (size_t i = 0; i < half_dim; ++i) {
        x[i] = 1.0f;
        x[i + half_dim] = 0.5f;

        y[i] = 2.0f;
        y[i + half_dim] = 3.0f;
    }

    // Expected pairing for each index: (1.0 * 3.0 - 0.5 * 2.0) = 3.0 - 1.0 = 2.0
    // Total for 32 indices = 64.0
    float pairing = weft::PELShimuraEngine::compute_symplectic_pairing(x.data(), y.data(), half_dim);
    std::cout << "  Symplectic pairing result: " << pairing << " (expected 64.0)" << std::endl;
    assert(std::fabs(pairing - 64.0f) < 1e-4f);

    // Self-pairing of any vector in symplectic space must be identically 0
    float self_pairing = weft::PELShimuraEngine::compute_symplectic_pairing(x.data(), x.data(), half_dim);
    std::cout << "  Self symplectic pairing: " << self_pairing << " (expected 0.0)" << std::endl;
    assert(std::fabs(self_pairing) < 1e-5f);

    std::cout << "[PASS] test_motivic_pel_shimura succeeded." << std::endl;
    return 0;
}
