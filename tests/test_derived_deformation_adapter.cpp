// test_derived_deformation_adapter.cpp
// Milestone Paper 237: Verification of AVX-512 Derived Deformation Theory & Adapter Merging
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_derived_deformation_adapter.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_adapter_fusion_kernel() {
    const size_t N = 32;
    std::vector<float> a(N, 2.0f);
    std::vector<float> b(N, 3.0f);
    std::vector<float> out(N, 0.0f);

    weft::DerivedAdapterMergeConfig cfg{0.6f, 0.4f, 0.12f};
    weft::avx512_derived_bch_adapter_fuse(a.data(), b.data(), out.data(), N, cfg);

    float expected = 0.6f * 2.0f + 0.4f * 3.0f; // 1.2 + 1.2 = 2.4
    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(out[i] - expected) < 1e-5f);
    }
}

int main() {
    std::cout << "[Paper 237] Running Derived Deformation Adapter Merging Unit Tests...\n";
    test_adapter_fusion_kernel();
    std::cout << "[Paper 237] All unit tests PASSED successfully.\n";
    return 0;
}
