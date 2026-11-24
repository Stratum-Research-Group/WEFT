// test_motivic_singular_kv_compaction.cpp
// Milestone Paper 242: Verification of AVX-512 Motivic Singular KV Compaction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_motivic_singular_kv_compaction.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_kv_compaction_kernel() {
    const size_t N = 32;
    std::vector<float> persistence(N, 0.0f);
    // Set 8 tokens with high persistence >= 10.0
    for (size_t i = 0; i < 8; ++i) {
        persistence[i * 4] = 15.0f;
    }

    std::vector<size_t> retained_indices(N, 0);
    size_t count = weft::avx512_compact_kv_by_persistence(persistence.data(), N, 10.0f, retained_indices.data());

    assert(count == 8);
    for (size_t i = 0; i < 8; ++i) {
        assert(retained_indices[i] == i * 4);
    }
}

int main() {
    std::cout << "[Paper 242] Running Motivic Singular KV Compaction Unit Tests...\n";
    test_kv_compaction_kernel();
    std::cout << "[Paper 242] All unit tests PASSED successfully.\n";
    return 0;
}
