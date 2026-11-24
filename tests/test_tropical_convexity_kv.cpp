// test_tropical_convexity_kv.cpp
// Milestone Paper 232: Verification of AVX-512 Tropical Convexity for KV Partitioning
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_convexity_kv.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_partition_disjointness() {
    weft::TropicalTaskPartition p1{1, 0, 1024, 4096};
    weft::TropicalTaskPartition p2{2, 1024, 2048, 4096};
    weft::TropicalTaskPartition p3{3, 512, 1024, 4096};

    // p1 and p2 are contiguous and disjoint
    assert(p1.is_disjoint_from(p2));
    assert(p2.is_disjoint_from(p1));

    // p1 and p3 overlap
    assert(!p1.is_disjoint_from(p3));
}

void test_avx512_max_plus_kernel() {
    const size_t N = 64;
    std::vector<float> a(N, 2.0f);
    std::vector<float> b(N, 4.0f);
    std::vector<float> out(N, 0.0f);

    float off_a = 5.0f; // 2 + 5 = 7
    float off_b = 1.0f; // 4 + 1 = 5

    weft::avx512_tropical_max_plus(a.data(), b.data(), out.data(), N, off_a, off_b);

    for (size_t i = 0; i < N; ++i) {
        // max(7, 5) = 7
        assert(std::fabs(out[i] - 7.0f) < 1e-5f);
    }
}

int main() {
    std::cout << "[Paper 232] Running Tropical Convexity KV Partitioning Unit Tests...\n";
    test_partition_disjointness();
    test_avx512_max_plus_kernel();
    std::cout << "[Paper 232] All unit tests PASSED successfully.\n";
    return 0;
}
