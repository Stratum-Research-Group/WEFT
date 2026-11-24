// test_derived_noncommutative_kv_dedup.cpp
// Milestone Paper 243: Verification of AVX-512 Derived Non-Commutative KV Deduplication
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_derived_noncommutative_kv_dedup.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_cyclic_trace_and_dedup() {
    const size_t N = 32;
    std::vector<float> a(N, 1.5f);
    std::vector<float> b(N, 2.0f);

    float trace = weft::avx512_compute_cyclic_trace(a.data(), b.data(), N);
    // Expected trace = 32 * (1.5 * 2.0) = 32 * 3.0 = 96.0
    assert(std::fabs(trace - 96.0f) < 1e-4f);

    // Test duplicate detection
    std::vector<float> cand_a = a;
    std::vector<float> cand_b = b;
    bool is_dup = weft::is_kv_block_duplicate(cand_a.data(), cand_b.data(), a.data(), b.data(), N, 0.01f);
    assert(is_dup);

    // Modify candidate
    cand_a[0] = 10.0f;
    bool is_not_dup = weft::is_kv_block_duplicate(cand_a.data(), cand_b.data(), a.data(), b.data(), N, 0.01f);
    assert(!is_not_dup);
}

int main() {
    std::cout << "[Paper 243] Running Derived Non-Commutative KV Deduplication Unit Tests...\n";
    test_cyclic_trace_and_dedup();
    std::cout << "[Paper 243] All unit tests PASSED successfully.\n";
    return 0;
}
