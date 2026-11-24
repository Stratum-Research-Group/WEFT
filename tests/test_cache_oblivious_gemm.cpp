// test_cache_oblivious_gemm.cpp - Verification for Paper #133
#include "csrc/cli/weft_cache_oblivious_gemm.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_cache_oblivious_gemm..." << std::endl;

    constexpr size_t M = 16;
    constexpr size_t N = 32;
    constexpr size_t K = 32;

    std::vector<float> A(M * K, 1.0f);
    // Packed 2-bit weights: K * (N / 4) bytes
    std::vector<uint8_t> W_packed(K * (N / 4));
    for (size_t i = 0; i < W_packed.size(); ++i) {
        // Discrete state 2 across all coordinates: 0b10101010
        W_packed[i] = 0b10101010;
    }

    std::vector<float> C(M * N, 0.0f);

    float scale = 0.5f;
    float zero_point = 0.0f;
    // Expected weight per coordinate: 2 * 0.5 + 0.0 = 1.0
    // Expected dot product per element: sum_{k=0}^{31} (1.0 * 1.0) = 32.0

    WeftCacheObliviousGEMM::gemm_recursive(
        A.data(),
        W_packed.data(),
        C.data(),
        M, N, K,
        /*lda=*/K, /*ldw=*/N / 4, /*ldc=*/N,
        scale, zero_point
    );

    std::cout << "[TEST] Output C[0, 0] = " << C[0] << " | C[M-1, N-1] = " << C[M * N - 1] << " (Expected 32.0)" << std::endl;

    for (size_t i = 0; i < M * N; ++i) {
        assert(std::abs(C[i] - 32.0f) < 1e-4f);
    }

    std::cout << "[TEST] All cache-oblivious recursive GEMM checks passed successfully!" << std::endl;
    return 0;
}
