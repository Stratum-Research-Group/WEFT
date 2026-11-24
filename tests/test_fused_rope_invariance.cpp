// test_fused_rope_invariance.cpp - Verification for Paper #129
#include "csrc/cli/weft_fused_rope.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_fused_rope_invariance..." << std::endl;

    WeftFusedRoPE rope_engine;
    std::vector<float> vec(WeftFusedRoPE::HEAD_DIM);

    // Initialize test vector
    for (size_t d = 0; d < WeftFusedRoPE::HEAD_DIM; ++d) {
        vec[d] = 1.0f + static_cast<float>(d) * 0.05f;
    }

    // Compute initial L2 norm
    float initial_norm_sq = 0.0f;
    for (size_t d = 0; d < WeftFusedRoPE::HEAD_DIM; ++d) {
        initial_norm_sq += vec[d] * vec[d];
    }
    float initial_norm = std::sqrt(initial_norm_sq);
    std::cout << "[TEST] Initial vector L2 norm: " << initial_norm << std::endl;

    // Apply RoPE at varied positions
    const std::vector<size_t> test_positions = {0, 1, 16, 512, 4096, 32768, 65535};
    for (size_t pos : test_positions) {
        std::vector<float> v_rot = vec;
        rope_engine.apply_rope(v_rot.data(), pos);

        float rot_norm_sq = 0.0f;
        for (size_t d = 0; d < WeftFusedRoPE::HEAD_DIM; ++d) {
            rot_norm_sq += v_rot[d] * v_rot[d];
        }
        float rot_norm = std::sqrt(rot_norm_sq);

        float norm_diff = std::abs(rot_norm - initial_norm);
        std::cout << "[TEST] Position " << pos << " | Rotated Norm: " << rot_norm << " | Delta: " << norm_diff << std::endl;

        // Unitary isometry guarantee: delta must be within floating-point epsilon
        assert(norm_diff < 1e-4f);
    }

    std::cout << "[TEST] Unitary isometry (norm preservation) verified across all sequence positions!" << std::endl;
    return 0;
}
