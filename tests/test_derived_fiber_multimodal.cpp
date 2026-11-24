// test_derived_fiber_multimodal.cpp
// Milestone Paper 249: Verification of AVX-512 Derived Fiber Multimodal Fusion
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_derived_fiber_multimodal.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_multimodal_fusion_kernel() {
    const size_t N = 32;
    std::vector<float> v(N, 2.0f);
    std::vector<float> t(N, 3.0f);
    std::vector<float> out(N, 0.0f);

    weft::DerivedMultimodalConfig cfg{0.5f, 0.5f, 0.1f};
    weft::avx512_derived_multimodal_fuse(v.data(), t.data(), out.data(), N, cfg);

    // Uniform vectors: v_curr * t_next - v_next * t_curr = 2 * 3 - 2 * 3 = 0.
    // fused = 0.5 * 2 + 0.5 * 3 = 1 + 1.5 = 2.5
    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(out[i] - 2.5f) < 1e-4f);
    }

    // Non-uniform test
    v[0] = 4.0f;
    weft::avx512_derived_multimodal_fuse(v.data(), t.data(), out.data(), N, cfg);
    // For i = 0: v0 = 4, t0 = 3, v1 = 2, t1 = 3.
    // tor1 = 4 * 3 - 2 * 3 = 12 - 6 = 6.
    // out[0] = 0.5 * 4 + 0.5 * 3 + 0.1 * 6 = 2 + 1.5 + 0.6 = 4.1
    assert(std::fabs(out[0] - 4.1f) < 1e-4f);
}

int main() {
    std::cout << "[Paper 249] Running Derived Fiber Multimodal Fusion Unit Tests...\n";
    test_multimodal_fusion_kernel();
    std::cout << "[Paper 249] All unit tests PASSED successfully.\n";
    return 0;
}
