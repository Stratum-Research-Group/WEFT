#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_subbit_bitplane_matmul.h"
#include "csrc/cli/weft_subbit_pack_engine.h"

int main() {
    std::cout << "[Test] Initializing Sub-Bit Bitplane MatMul SIMD Test..." << std::endl;

    const size_t count = 32;
    std::vector<float> activations(count, 1.5f);
    std::vector<float> weights(count, 0.0f);

    float expected_dot = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        activations[i] = static_cast<float>(i + 1) * 0.5f;
        if (i % 3 == 0) weights[i] = 1.0f;
        else if (i % 3 == 1) weights[i] = -1.0f;
        else weights[i] = 0.0f;

        expected_dot += activations[i] * weights[i];
    }

    std::vector<uint8_t> packed_weights(count / 4, 0);
    weft::avx512_pack_ternary_subbit(weights.data(), packed_weights.data(), count);

    float computed_dot = weft::avx512_subbit_dot_product(
        activations.data(),
        packed_weights.data(),
        count
    );

    std::cout << "[Test] Expected Dot Product: " << expected_dot << std::endl;
    std::cout << "[Test] Computed Dot Product: " << computed_dot << std::endl;
    assert(std::fabs(computed_dot - expected_dot) < 1e-4f);

    std::cout << "[Test] Sub-Bit Bitplane MatMul SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
