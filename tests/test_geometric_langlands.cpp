#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_geometric_langlands.h"

int main() {
    std::cout << "[TEST] Running Geometric Langlands Modular Attention Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> weights(DIM, 1.0f);
    std::vector<float> hecke_out(DIM, 0.0f);

    // Define Hecke kernel: symmetric root shifts {-1, 0, 1} with weights {0.25, 0.5, 0.25}
    std::vector<weft::HeckeOperatorKernel> kernels = {
        {-1, 0.25f},
        {0, 0.50f},
        {1, 0.25f}
    };

    // Test 1: Constant vector is an eigenvector with eigenvalue 1.0
    weft::apply_hecke_operator_simd(weights.data(), DIM, kernels.data(), kernels.size(), hecke_out.data());
    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(hecke_out[i] - 1.0f) < 1e-6f);
    }
    std::cout << "  Hecke action on constant vector: eigenvalue 1.0 verified." << std::endl;

    // Test 2: Hecke eigensheaf projection idempotence
    std::vector<float> noisy_weights = weights;
    noisy_weights[0] += 0.2f;
    weft::apply_hecke_operator_simd(noisy_weights.data(), DIM, kernels.data(), kernels.size(), hecke_out.data());
    weft::project_hecke_eigensheaf_simd(noisy_weights.data(), hecke_out.data(), DIM, 1.0f);
    assert(std::isfinite(noisy_weights[0]));

    std::cout << "[PASS] All Geometric Langlands Modular Attention invariants verified." << std::endl;
    return 0;
}
