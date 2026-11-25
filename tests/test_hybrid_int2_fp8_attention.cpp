// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_hybrid_int2_fp8_attention.h"

int main() {
    std::cout << "[TEST] Running Hybrid INT2/FP8 Attention AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> q(dim);
    std::vector<uint8_t> k_packed(dim / 4, 0);
    float k_scale = 0.5f;

    for (size_t i = 0; i < dim; ++i) {
        q[i] = 1.0f + 0.1f * (i % 8);
    }

    // Set packed pattern: 01 (1.0), 10 (-1.0), 00 (0.0), 01 (1.0) -> byte = 0b01001001 = 0x49
    for (size_t b = 0; b < dim / 4; ++b) {
        k_packed[b] = 0x49; // bit0: 1, bit1: 2, bit2: 0, bit3: 1
    }

    float dot = weft::HybridInt2Fp8Attention::compute_dot_product(q.data(), k_packed.data(), k_scale, dim);

    float expected = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        size_t bit = i % 4;
        uint8_t code = (0x49 >> (bit * 2)) & 0x03;
        float val = 0.0f;
        if (code == 1) val = 1.0f;
        else if (code == 2) val = -1.0f;
        expected += q[i] * (val * k_scale);
    }

    assert(std::abs(dot - expected) < 1e-4f);
    std::cout << "[PASS] Hybrid INT2/FP8 Attention kernel verified successfully." << std::endl;
    return 0;
}
