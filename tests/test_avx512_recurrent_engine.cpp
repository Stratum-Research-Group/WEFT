#include "../csrc/kernels/avx512_recurrent_engine.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "[TEST] Running AVX-512 In-Register Recurrent Engine Test..." << std::endl;
    constexpr size_t R = 8;
    constexpr size_t D = 64;

    std::vector<float> state(R * D, 0.0f);
    std::vector<float> gamma(R, 0.0f); // sigma(0) = 0.5
    std::vector<float> k(R, 1.0f);
    std::vector<float> v(D, 2.0f);

    // Step 1: S_1 = 0.5 * 0 + 1.0 * 2.0 = 2.0
    weft::avx512_recurrent_step(state.data(), gamma.data(), k.data(), v.data(), R, D);
    for (size_t i = 0; i < R * D; ++i) {
        assert(std::fabs(state[i] - 2.0f) < 1e-5f);
    }

    // Step 2: S_2 = 0.5 * 2.0 + 1.0 * 2.0 = 3.0
    weft::avx512_recurrent_step(state.data(), gamma.data(), k.data(), v.data(), R, D);
    for (size_t i = 0; i < R * D; ++i) {
        assert(std::fabs(state[i] - 3.0f) < 1e-5f);
    }

    std::cout << "[DONE] AVX-512 In-Register Recurrent Engine passed successfully." << std::endl;
    return 0;
}
