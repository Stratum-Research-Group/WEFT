// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Tropical Jacobian Attention Routing Invariants (Paper 229)

#include "csrc/cli/weft_tropical_jacobian.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_route_tokens_tropical() {
    const size_t tokens = 4;
    const size_t experts = 4;
    std::vector<float> token_p = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expert_p = {10.0f, 20.0f, 5.0f, 15.0f};
    std::vector<uint32_t> selected(tokens, 0);

    weft::tropical_jacobian::route_tokens_tropical_avx512(
        token_p.data(), expert_p.data(), selected.data(), tokens, experts
    );

    // Expert 1 has maximum potential (20.0), so all tokens should route to expert 1
    for (size_t t = 0; t < tokens; ++t) {
        assert(selected[t] == 1);
    }
    std::cout << "[PASS] test_route_tokens_tropical (all tokens route to expert 1)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 229: Tropical Jacobian Routing Unit Tests ===" << std::endl;
    test_route_tokens_tropical();
    std::cout << "All Tropical Jacobian Routing tests PASSED." << std::endl;
    return 0;
}
