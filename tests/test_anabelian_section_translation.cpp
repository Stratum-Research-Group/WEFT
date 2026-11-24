// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Anabelian Section Architecture Translation Invariants (Paper 227)

#include "csrc/cli/weft_anabelian_section_translation.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_translate_architecture_weights() {
    const size_t N = 32;
    std::vector<float> src(N);
    std::vector<uint32_t> perm(N);
    std::vector<float> dst(N, 0.0f);

    for (size_t i = 0; i < N; ++i) {
        src[i] = static_cast<float>(i * 5);
        perm[i] = static_cast<uint32_t>((i + 7) % N);
    }

    float scale = 2.0f;
    weft::anabelian_translation::translate_architecture_weights_avx512(
        src.data(), perm.data(), dst.data(), N, scale
    );

    for (size_t i = 0; i < N; ++i) {
        float expected = src[perm[i]] * scale;
        assert(std::abs(dst[i] - expected) < 1e-5f);
    }
    std::cout << "[PASS] test_translate_architecture_weights (N=32, scale 2.0 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 227: Anabelian Architecture Translation Unit Tests ===" << std::endl;
    test_translate_architecture_weights();
    std::cout << "All Anabelian Architecture Translation tests PASSED." << std::endl;
    return 0;
}
