// test_tropical_speculation.cpp
// Validation of Tropical Min-Plus matrix-vector product and candidate path selection.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_speculation.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_tropical_speculation..." << std::endl;

    const int rows = 4;
    const int cols = 16;
    std::vector<float> A(rows * cols);
    std::vector<float> x(cols);

    // Costs: x_j = float(j) * 0.5f
    for (int j = 0; j < cols; ++j) {
        x[j] = float(j) * 0.5f;
    }

    // Set A matrix so row 2 has minimum cost path through col 3:
    // A_{r, j} = 10.0f except A_{2, 3} = 1.0f
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            A[r * cols + c] = 10.0f;
        }
    }
    A[2 * cols + 3] = 1.0f; // min for row 2 should be 1.0 + 3*0.5 = 2.5

    std::vector<float> out(rows);
    weft::tropical_gemv_avx512(A.data(), x.data(), out.data(), rows, cols);

    std::cout << "[TEST] Tropical GEMV row[2] = " << out[2] << " (Expected 2.5)" << std::endl;
    assert(std::abs(out[2] - 2.5f) < 1e-5f);

    int best_cand = weft::tropical_best_candidate(out.data(), rows);
    std::cout << "[TEST] Best Candidate Row: " << best_cand << " (Expected 2)" << std::endl;
    assert(best_cand == 2);

    std::cout << "[TEST] All Tropical Semiring Speculation tests passed successfully!" << std::endl;
    return 0;
}
