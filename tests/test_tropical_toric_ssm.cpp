#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_tropical_toric_ssm.h"

int main() {
    std::cout << "[Test] Initializing Tropical Toric SSM Recurrence SIMD Test..." << std::endl;

    const size_t count = 32;
    const float a = 0.5f;
    const float b = 1.0f;

    std::vector<float> hidden(count, 0.0f);
    std::vector<float> input(count, 0.0f);
    std::vector<float> out_state(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        hidden[i] = static_cast<float>(i);
        input[i] = static_cast<float>(count - i);
    }

    weft::avx512_tropical_ssm_step(
        hidden.data(),
        input.data(),
        out_state.data(),
        count,
        a,
        b
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = std::max(hidden[i] + a, input[i] + b);
        assert(std::fabs(out_state[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] State [0]: " << out_state[0] << std::endl;
    std::cout << "[Test] State [31]: " << out_state[31] << std::endl;
    std::cout << "[Test] Tropical Toric SSM Recurrence SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
