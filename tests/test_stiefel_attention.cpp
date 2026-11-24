// test_stiefel_attention.cpp
// Validation of Stiefel manifold column orthonormality and Cayley retraction.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_stiefel_attention.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_stiefel_attention..." << std::endl;

    const int d = 64; // Embedding dimension
    const int k = 4;  // Number of orthonormal columns

    weft::StiefelMatrix X(d, k);
    // Initialize canonical orthonormal basis columns
    for (int c = 0; c < k; ++c) {
        X.at(c, c) = 1.0f;
    }

    // Check initial Gram matrix X^T X = I
    for (int c1 = 0; c1 < k; ++c1) {
        for (int c2 = 0; c2 < k; ++c2) {
            float dot = weft::column_dot_avx512(X, c1, c2);
            float expected = (c1 == c2) ? 1.0f : 0.0f;
            assert(std::abs(dot - expected) < 1e-6f);
        }
    }
    std::cout << "[TEST] Initial Orthonormality Confirmed." << std::endl;

    // Create a tangent matrix Delta
    weft::StiefelMatrix Delta(d, k);
    for (int c = 0; c < k; ++c) {
        Delta.at((c + 10) % d, c) = 0.5f;
    }

    // Perform Cayley step
    weft::stiefel_cayley_step(X, Delta, 0.1f);

    // Verify post-step column unit norms
    for (int c = 0; c < k; ++c) {
        float norm_sq = weft::column_dot_avx512(X, c, c);
        std::cout << "[TEST] Column " << c << " Norm Squared: " << norm_sq << std::endl;
        assert(std::abs(norm_sq - 1.0f) < 1e-5f);
    }

    std::cout << "[TEST] All Stiefel Manifold Attention tests passed successfully!" << std::endl;
    return 0;
}
