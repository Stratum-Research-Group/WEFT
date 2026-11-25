#include "../csrc/cli/weft_derived_motivic_hodge.h"
#include "../csrc/cli/weft_tropical_hyperplanes.h"
#include "../csrc/cli/weft_motivic_galois_residual.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "=== Test Papers 318-320 C++20 Kernels ===" << std::endl;

    // 1. Paper 318: Motivic Hodge
    assert(weft::MotivicHodgeQuantizer::truncation_error(100, 80) == 20);
    std::cout << "[PASS] Paper 318 Derived Motivic Hodge Kernel." << std::endl;

    // 2. Paper 319: Tropical Hyperplanes
    std::vector<float> x = {1.0f, 4.0f, 2.0f};
    std::vector<float> c = {3.0f, 1.0f, 6.0f};
    float max_h = weft::TropicalHyperplanesEngine::tropical_hyperplane_eval(x.data(), c.data(), 3);
    assert(std::abs(max_h - 8.0f) < 1e-5f);
    std::cout << "[PASS] Paper 319 Tropical Hyperplanes Kernel." << std::endl;

    // 3. Paper 320: Motivic Galois Residual
    float res_trace = weft::MotivicGaloisResidualEngine::residual_accumulate(10.5f, 2.5f);
    assert(std::abs(res_trace - 13.0f) < 1e-5f);
    std::cout << "[PASS] Paper 320 Motivic Galois Residual Kernel." << std::endl;

    return 0;
}
