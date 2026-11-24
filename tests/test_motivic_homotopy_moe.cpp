// test_motivic_homotopy_moe.cpp
// Milestone Paper 234: Verification of AVX-512 Motivic Homotopy MoE Routing & Retraction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_motivic_homotopy_moe.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_hopf_projection_kernel() {
    const size_t num_experts = 32;
    std::vector<float> logits(num_experts, 1.0f);
    std::vector<float> projected(num_experts, 0.0f);

    float lambda = 0.01f;
    // sq_sum = 32 * (1^2) = 32. denom = 1 + 0.01 * 32 = 1.32. scale = 1 / 1.32 = 0.757575
    weft::avx512_motivic_hopf_project(logits.data(), projected.data(), num_experts, lambda);

    float expected = 1.0f / (1.0f + 0.01f * 32.0f);
    for (size_t i = 0; i < num_experts; ++i) {
        assert(std::fabs(projected[i] - expected) < 1e-4f);
    }
}

void test_topk_selection() {
    const size_t num_experts = 16;
    const size_t top_k = 4;
    std::vector<float> scores(num_experts, 0.0f);
    scores[3] = 10.0f;
    scores[7] = 8.0f;
    scores[11] = 6.0f;
    scores[1] = 4.0f;

    std::vector<size_t> top_indices(top_k, 0);
    weft::select_topk_experts(scores.data(), num_experts, top_k, top_indices.data());

    assert(top_indices[0] == 3);
    assert(top_indices[1] == 7);
    assert(top_indices[2] == 11);
    assert(top_indices[3] == 1);
}

int main() {
    std::cout << "[Paper 234] Running Motivic Homotopy MoE Routing Unit Tests...\n";
    test_hopf_projection_kernel();
    test_topk_selection();
    std::cout << "[Paper 234] All unit tests PASSED successfully.\n";
    return 0;
}
