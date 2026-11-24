// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Liquid Vector Bundle KV Cache Virtualization Invariants (Paper 225)

#include "csrc/cli/weft_liquid_kv_virtualization.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_batch_page_lookup() {
    const size_t N = 32;
    std::vector<uint32_t> tokens(N);
    std::vector<uint32_t> pages(N, 0);
    std::vector<uint32_t> offsets(N, 0);

    for (size_t i = 0; i < N; ++i) {
        tokens[i] = static_cast<uint32_t>(i);
    }

    uint32_t page_shift = 4; // 16 tokens per page
    weft::liquid_kv_virtualization::batch_virtual_page_lookup_avx512(
        tokens.data(), pages.data(), offsets.data(), N, page_shift
    );

    for (size_t i = 0; i < N; ++i) {
        assert(pages[i] == (tokens[i] >> 4));
        assert(offsets[i] == (tokens[i] & 15));
        assert(pages[i] * 16 + offsets[i] == tokens[i]);
    }
    std::cout << "[PASS] test_batch_page_lookup (N=32, page indexing verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 225: Liquid KV Virtualization Unit Tests ===" << std::endl;
    test_batch_page_lookup();
    std::cout << "All Liquid KV Virtualization tests PASSED." << std::endl;
    return 0;
}
