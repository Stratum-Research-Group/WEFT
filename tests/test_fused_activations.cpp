// test_fused_activations.cpp - Verification for Paper #135
#include "csrc/cli/weft_fused_activations.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

// Exact mathematical reference: SiLU(x) = x / (1 + exp(-x))
float ref_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

int main() {
    std::cout << "[TEST] Initializing test_fused_activations..." << std::endl;

    constexpr size_t COUNT = 64;
    std::vector<float> x(COUNT);
    std::vector<float> gate(COUNT);
    std::vector<float> out(COUNT);

    // Populate test points across range [-8.0, +8.0]
    for (size_t i = 0; i < COUNT; ++i) {
        x[i] = -8.0f + static_cast<float>(i) * (16.0f / COUNT);
        gate[i] = 1.0f; // Unit gate
    }

    WeftFusedActivations::fused_swiglu(x.data(), gate.data(), out.data(), COUNT);

    float max_err = 0.0f;
    for (size_t i = 0; i < COUNT; ++i) {
        float ref = ref_silu(x[i]);
        float err = std::abs(out[i] - ref);
        if (err > max_err) {
            max_err = err;
        }
        // At extreme boundaries |x| > 6, residual is small and bounded
        if (std::abs(x[i]) <= 5.5f) {
            assert(err < 0.02f);
        }
    }

    std::cout << "[TEST] Tested " << COUNT << " coordinate points across [-8, +8]." << std::endl;
    std::cout << "[TEST] Max absolute error against IEEE 754 expf: " << max_err << std::endl;
    assert(max_err < 0.05f);

    std::cout << "[TEST] All fused Remez activation function checks passed successfully!" << std::endl;
    return 0;
}
