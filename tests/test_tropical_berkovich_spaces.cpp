#include "csrc/cli/weft_tropical_berkovich_spaces.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t half_dim = 64;
    std::vector<float> base_inv_freq(half_dim);
    std::vector<float> cos_out(half_dim, 0.0f);
    std::vector<float> sin_out(half_dim, 0.0f);

    for (size_t i = 0; i < half_dim; ++i) {
        base_inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(half_dim * 2));
    }

    uint32_t token_pos = 131072; // 128k token position
    weft::tropical_berkovich_rope_phases_avx512(
        base_inv_freq.data(), cos_out.data(), sin_out.data(),
        half_dim, token_pos, 4.0f);

    for (size_t i = 0; i < half_dim; ++i) {
        assert(!std::isnan(cos_out[i]));
        assert(!std::isinf(cos_out[i]));
        assert(!std::isnan(sin_out[i]));
        assert(!std::isinf(sin_out[i]));
    }

    std::cout << "[PASS] Tropical Berkovich Spaces RoPE Skeletal Extrapolator Unit Test passed.\n";
    return 0;
}
