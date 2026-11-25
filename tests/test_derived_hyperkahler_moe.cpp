// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_derived_hyperkahler_moe.h"

int main() {
    std::cout << "[TEST] Running Hyperkähler MoE Routing Engine AVX-512 Test..." << std::endl;

    const size_t num_experts = 32;
    const size_t top_k = 2;
    std::vector<float> raw_logits(num_experts);
    std::vector<float> real_moment(num_experts, 0.0f);
    std::vector<float> complex_moment(num_experts, 0.0f);
    std::vector<float> projected_logits(num_experts, 0.0f);
    std::vector<size_t> out_experts(top_k);

    for (size_t i = 0; i < num_experts; ++i) {
        raw_logits[i] = (float)i * 0.1f;
        real_moment[i] = (i == 31) ? 20.0f : 0.0f; // Penalize oscillating expert 31
    }

    weft::HyperkahlerMoERoutingEngine::regularize_routing_logits(
        raw_logits.data(), real_moment.data(), complex_moment.data(),
        projected_logits.data(), num_experts, 0.2f
    );

    weft::HyperkahlerMoERoutingEngine::select_top_k(
        projected_logits.data(), num_experts, top_k, out_experts.data()
    );

    // Expert 31 was penalized: 3.1 - 0.2*20 = -0.9, so top experts should be 30 and 29
    assert(out_experts[0] == 30);
    assert(out_experts[1] == 29);

    std::cout << "[PASS] Hyperkähler MoE Routing Engine verified successfully (Top experts: "
              << out_experts[0] << ", " << out_experts[1] << ")." << std::endl;
    return 0;
}
