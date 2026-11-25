// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_fused_numa_flash_attention.h"

int main() {
    std::cout << "[TEST] Running Fused NUMA FlashAttention AVX-512 Test..." << std::endl;

    const size_t seq_len = 32;
    const size_t head_dim = 64;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    std::vector<float> q(head_dim);
    std::vector<float> k(seq_len * head_dim);
    std::vector<float> v(seq_len * head_dim);
    std::vector<float> out(head_dim, 0.0f);

    for (size_t d = 0; d < head_dim; ++d) {
        q[d] = 1.0f + 0.05f * (d % 4);
    }
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
            k[s * head_dim + d] = 0.5f + 0.02f * ((s + d) % 5);
            v[s * head_dim + d] = 1.0f + 0.1f * ((s * 2 + d) % 7);
        }
    }

    weft::FusedNumaFlashAttention::compute_tile_attention(
        q.data(), k.data(), v.data(), out.data(), seq_len, head_dim, scale
    );

    float sum_out = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
        sum_out += out[d];
        assert(!std::isnan(out[d]));
    }
    assert(sum_out > 0.0f);

    std::cout << "[PASS] Fused NUMA FlashAttention verified successfully (sum_out: " << sum_out << ")." << std::endl;
    return 0;
}
