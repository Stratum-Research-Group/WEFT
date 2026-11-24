// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Motivic Cohomological Hall Algebra Invariants (Paper 224)

#include "csrc/cli/weft_motivic_cohall.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_generate_coha_embedding() {
    const size_t dim = 32;
    std::vector<float> basis_a(dim, 2.0f);
    std::vector<float> basis_b(dim, 3.0f);
    std::vector<float> token_out(dim, 0.0f);

    float alpha = 0.5f;
    float beta = 0.5f;

    // expected = 2.0*0.5 + 3.0*0.5 + 2.0*3.0 = 1.0 + 1.5 + 6.0 = 8.5
    weft::motivic_cohall::generate_coha_embedding_avx512(
        basis_a.data(), basis_b.data(), token_out.data(), dim, alpha, beta
    );

    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(token_out[i] - 8.5f) < 1e-5f);
    }
    std::cout << "[PASS] test_generate_coha_embedding (dim=32, value 8.5 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 224: Motivic COHA Unit Tests ===" << std::endl;
    test_generate_coha_embedding();
    std::cout << "All Motivic COHA tests PASSED." << std::endl;
    return 0;
}
