#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_motivic_galois_alignment.h"

int main() {
    std::cout << "[Test] Initializing Motivic Galois Alignment SIMD Test..." << std::endl;

    const size_t count = 32;
    const float w1 = 0.6f;
    const float w2 = 0.4f;

    std::vector<float> mod1(count, 0.0f);
    std::vector<float> mod2(count, 0.0f);
    std::vector<float> out_aligned(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        mod1[i] = static_cast<float>(i + 1);
        mod2[i] = static_cast<float>(count - i);
    }

    weft::avx512_tannakian_alignment_blend(
        mod1.data(),
        mod2.data(),
        out_aligned.data(),
        count,
        w1,
        w2
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = w1 * mod1[i] + w2 * mod2[i];
        assert(std::fabs(out_aligned[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Aligned [0]: " << out_aligned[0] << std::endl;
    std::cout << "[Test] Aligned [31]: " << out_aligned[31] << std::endl;
    std::cout << "[Test] Motivic Galois Alignment SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
