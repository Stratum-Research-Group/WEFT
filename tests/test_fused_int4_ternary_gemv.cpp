#include "csrc/cli/weft_fused_int4_ternary_gemv.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim = 64;
    std::vector<uint8_t> act_int4(dim / 2, 0x88); // 8 - 8 = 0
    std::vector<int8_t> weights_ternary(dim, 1);  // all +1

    // Set nibble 0 to 12 (value: 12 - 8 = +4)
    act_int4[0] = 0x8C; // low nibble = 12, high nibble = 8

    float dot = weft::fused_int4_ternary_gemv_avx512(
        act_int4.data(), weights_ternary.data(), dim, 1.0f, 1.0f);

    // Expected dot = 4.0f * 1 * 1.0 = 4.0f
    assert(std::abs(dot - 4.0f) < 1e-4f);

    std::cout << "[PASS] Fused INT4 / Ternary GEMV Unit Test passed (Dot: " << dot << ").\n";
    return 0;
}
