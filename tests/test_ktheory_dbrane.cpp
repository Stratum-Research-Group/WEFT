#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_ktheory_dbrane.h"

int main() {
    std::cout << "[TEST] Running K-Theory D-Brane Charge Quantization Test..." << std::endl;

    constexpr float TWO_PI = 6.283185307179586f;

    // Test 1: Integer winding number evaluation
    int32_t q0 = weft::compute_topological_charge(0.2f);
    int32_t q1 = weft::compute_topological_charge(TWO_PI * 1.1f);
    int32_t q_neg = weft::compute_topological_charge(-TWO_PI * 1.9f);

    std::cout << "  Charges: q0=" << q0 << ", q1=" << q1 << ", q_neg=" << q_neg << std::endl;
    assert(q0 == 0);
    assert(q1 == 1);
    assert(q_neg == -2);

    // Test 2: In-register SIMD charge lattice snapping
    const size_t DIM = 32;
    std::vector<float> phases(DIM);
    std::vector<float> anchored(DIM);
    for (size_t i = 0; i < DIM; ++i) {
        phases[i] = TWO_PI * (static_cast<float>(i % 5) + 0.1f);
    }

    weft::snap_to_dbrane_charge_lattice_simd(phases.data(), DIM, anchored.data());

    for (size_t i = 0; i < DIM; ++i) {
        float expected = TWO_PI * static_cast<float>(i % 5);
        assert(std::abs(anchored[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] All K-Theory D-Brane Charge Quantization invariants verified." << std::endl;
    return 0;
}
