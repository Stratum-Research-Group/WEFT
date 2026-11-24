#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_tropical_toric.h"

int main() {
    std::cout << "[TEST] Running Tropical Toric Amoeba In-Register Quantization Test..." << std::endl;

    // Define tropical terms: f1(x) = 1.0 * x + 2.0, f2(x) = -1.0 * x + 4.0, f3(x) = 0.5 * x + 1.0
    std::vector<weft::TropicalAmoebaTerm> terms = {
        {1.0f, 2.0f},
        {-1.0f, 4.0f},
        {0.5f, 1.0f}
    };

    // Test 1: Scalar evaluation at x = 2.0
    // f1(2) = 4.0, f2(2) = 2.0, f3(2) = 2.0 -> min = 2.0
    float scalar_res = weft::eval_tropical_ronkin(2.0f, terms.data(), terms.size());
    std::cout << "  Scalar tropical evaluation at x=2.0: " << scalar_res << std::endl;
    assert(std::abs(scalar_res - 2.0f) < 1e-5f);

    // Test 2: SIMD vector evaluation across logits
    const size_t N = 32;
    std::vector<float> logits(N);
    std::vector<float> output(N);
    for (size_t i = 0; i < N; ++i) {
        logits[i] = -2.0f + 0.2f * i;
    }

    weft::eval_tropical_amoeba_attention(logits.data(), N, terms.data(), terms.size(), output.data());

    for (size_t i = 0; i < N; ++i) {
        float expected = weft::eval_tropical_ronkin(logits[i], terms.data(), terms.size());
        assert(std::abs(output[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] All Tropical Toric Amoeba invariants verified." << std::endl;
    return 0;
}
