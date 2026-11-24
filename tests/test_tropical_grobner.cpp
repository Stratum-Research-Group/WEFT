// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Tropical Toric Ideals & Gröbner Bases Invariants (Paper 223)

#include "csrc/cli/weft_tropical_grobner.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_sparse_gemv_grobner() {
    const size_t dense_dim = 64;
    std::vector<float> dense_x(dense_dim);
    for (size_t i = 0; i < dense_dim; ++i) dense_x[i] = static_cast<float>(i);

    const size_t nonzeros = 32;
    std::vector<uint32_t> cols(nonzeros);
    std::vector<float> vals(nonzeros, 1.0f);

    float expected_sum = 0.0f;
    for (size_t i = 0; i < nonzeros; ++i) {
        cols[i] = static_cast<uint32_t>(i * 2);
        expected_sum += vals[i] * dense_x[cols[i]];
    }

    float out_accum = 0.0f;
    weft::tropical_grobner::sparse_gemv_grobner_avx512(
        dense_x.data(), cols.data(), vals.data(), nonzeros, &out_accum
    );

    assert(std::abs(out_accum - expected_sum) < 1e-4f);
    std::cout << "[PASS] test_sparse_gemv_grobner (sparse dot product verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 223: Tropical Toric Ideals Unit Tests ===" << std::endl;
    test_sparse_gemv_grobner();
    std::cout << "All Tropical Toric Ideals tests PASSED." << std::endl;
    return 0;
}
