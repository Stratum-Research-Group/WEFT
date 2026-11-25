#include "csrc/cli/weft_motivic_calabi_yau_4folds.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim_4bit = 16;
    constexpr size_t dim_ternary = 48;
    constexpr size_t dim_total = dim_4bit + dim_ternary;

    std::vector<float> query(dim_total, 1.0f);
    std::vector<uint8_t> key_int4(dim_4bit / 2, 0x88); // 8 - 8 = 0
    std::vector<int8_t> key_ternary(dim_ternary, 1);

    // Set first nibble to 10 (value: 10 - 8 = 2)
    key_int4[0] = 0x8A; // low nibble = 10, high nibble = 8

    float dot = weft::cy4_mixed_precision_dot_avx512(
        query.data(), key_int4.data(), key_ternary.data(),
        dim_4bit, dim_ternary, 0.125f, 0.5f);

    // Expected:
    // 4-bit dot: query[0] * 2 * 0.125f = 0.25f
    // Ternary dot: 48 * 1.0f * 1 * 0.5f = 24.0f
    // Total dot = 24.25f
    assert(std::abs(dot - 24.25f) < 1e-4f);

    std::cout << "[PASS] Motivic Calabi-Yau 4-Folds Mixed-Precision Attention Unit Test passed (Dot: " 
              << dot << ").\n";
    return 0;
}
