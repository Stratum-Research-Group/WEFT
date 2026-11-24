#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_motivic_cohomology.h"

int main() {
    std::cout << "[TEST] Running Motivic Cohomology & Beilinson Regulator Precision Routing Test..." << std::endl;

    // Test 1: Regulator scalar evaluation
    float r0 = weft::compute_beilinson_regulator_scalar(0.0f);
    float r1 = weft::compute_beilinson_regulator_scalar(std::exp(1.0f) - 1.0f);
    std::cout << "  Regulators: r(0)=" << r0 << ", r(e-1)=" << r1 << std::endl;
    assert(r0 == 0.0f);
    assert(std::abs(r1 - 1.0f) < 1e-5f);

    // Test 2: In-register precision allocation
    const size_t DIM = 32;
    std::vector<float> weights(DIM);
    std::vector<uint8_t> tags(DIM, 0);

    for (size_t i = 0; i < DIM; ++i) {
        weights[i] = (i % 2 == 0) ? 0.2f : 2.5f;
    }

    // tau_reg = log(1 + 1.0) = 0.6931f
    weft::MotivicPrecisionAllocation alloc = weft::route_motivic_precision_simd(
        weights.data(), DIM, 0.6931f, tags.data()
    );

    std::cout << "  High-prec count: " << alloc.high_prec_count << ", Low-prec count: " << alloc.low_prec_count << std::endl;
    assert(alloc.high_prec_count == 16);
    assert(alloc.low_prec_count == 16);
    assert(alloc.total_regulator_volume > 0.0f);

    std::cout << "[PASS] All Motivic Cohomology Precision Routing invariants verified." << std::endl;
    return 0;
}
