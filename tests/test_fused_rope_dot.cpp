#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_fused_rope_dot.h"

int main() {
    std::cout << "[Test] Initializing Fused RoPE Dot Product SIMD Test..." << std::endl;

    const size_t head_dim = 64;
    const float scale = 0.1f;

    std::vector<float> Q(head_dim, 1.0f);
    std::vector<float> Q_corr(head_dim, 0.5f);
    std::vector<float> K(head_dim, 2.0f);
    std::vector<float> K_corr(head_dim, 0.2f);

    for (size_t i = 0; i < head_dim; ++i) {
        Q[i] = static_cast<float>(i + 1) * 0.1f;
        Q_corr[i] = static_cast<float>(i % 4) * 0.05f;
        K[i] = static_cast<float>(head_dim - i) * 0.1f;
        K_corr[i] = static_cast<float>(i % 3) * 0.05f;
    }

    float dot_computed = weft::avx512_fused_rope_dot_product(
        Q.data(),
        Q_corr.data(),
        K.data(),
        K_corr.data(),
        head_dim,
        scale
    );

    float dot_expected = 0.0f;
    for (size_t i = 0; i < head_dim; ++i) {
        float q_rot = Q[i] - scale * Q_corr[i];
        float k_rot = K[i] - scale * K_corr[i];
        dot_expected += q_rot * k_rot;
    }

    std::cout << "[Test] Computed Fused Dot: " << dot_computed << ", Expected: " << dot_expected << std::endl;
    assert(std::fabs(dot_computed - dot_expected) < 1e-4f);

    std::cout << "[Test] Fused RoPE Dot Product SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
