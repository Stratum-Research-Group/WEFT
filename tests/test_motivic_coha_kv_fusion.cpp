#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_motivic_coha_kv_fusion.h"

int main() {
    std::cout << "[Test] Initializing Motivic COHA KV Fusion SIMD Test..." << std::endl;

    const size_t count = 32;
    std::vector<float> m1(count, 0.0f);
    std::vector<float> l1(count, 1.0f);
    std::vector<float> m2(count, 0.0f);
    std::vector<float> l2(count, 1.0f);
    std::vector<float> out_max(count, 0.0f);
    std::vector<float> out_sum(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        m1[i] = static_cast<float>(i % 8);
        m2[i] = static_cast<float>((i + 3) % 8);
    }

    weft::avx512_coha_kv_fusion(
        m1.data(),
        l1.data(),
        m2.data(),
        l2.data(),
        out_max.data(),
        out_sum.data(),
        count
    );

    for (size_t i = 0; i < count; ++i) {
        float expected_max = std::max(m1[i], m2[i]);
        assert(std::fabs(out_max[i] - expected_max) < 1e-5f);
        assert(out_sum[i] > 0.0f);
    }

    std::cout << "[Test] Fused Tile [0] max logit: " << out_max[0] << ", sum: " << out_sum[0] << std::endl;
    std::cout << "[Test] Motivic COHA KV Fusion SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
