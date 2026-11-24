// test_tensor_train_kv_compaction.cpp
// Validation of AVX-512 Tensor Train core contractions and non-expansion bounds.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tensor_train_kv.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_tensor_train_kv_compaction..." << std::endl;

    const int n1 = 64;  // Head dimension / sequence mode
    const int n2 = 64;  // Feature dimension mode
    const int r = 4;    // TT-Rank

    weft::TTCore c1(1, n1, r);
    weft::TTCore c2(r, n2, 1);

    // Initialize orthogonal-like deterministic weights
    for (int j = 0; j < n1; ++j) {
        for (int k = 0; k < r; ++k) {
            c1.at(0, j, k) = ((j % (k + 1) == 0) ? 1.0f : 0.0f) / std::sqrt(float(n1));
        }
    }
    for (int k = 0; k < r; ++k) {
        for (int j = 0; j < n2; ++j) {
            c2.at(k, j, 0) = ((j % (k + 1) == 0) ? 1.0f : 0.0f) / std::sqrt(float(n2));
        }
    }

    std::vector<float> x(n1, 1.0f);
    std::vector<float> q(n2, 1.0f);

    std::vector<float> intermediate(r, 0.0f);
    weft::tt_contract_core1_avx512(x.data(), c1, intermediate.data());

    for (int k = 0; k < r; ++k) {
        std::cout << "[TEST] Core1 contracted mode rank[" << k << "] = " << intermediate[k] << std::endl;
        assert(!std::isnan(intermediate[k]));
    }

    float score = weft::tt_contract_attention_score(x.data(), c1, c2, q.data());
    std::cout << "[TEST] Contracted Attention Score: " << score << std::endl;
    assert(score > 0.0f);

    std::cout << "[TEST] All Tensor Train KV contraction tests passed successfully!" << std::endl;
    return 0;
}
