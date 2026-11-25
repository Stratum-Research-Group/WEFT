// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_derived_azumaya_algebra.h"

int main() {
    std::cout << "[TEST] Running Azumaya Secret Share Engine AVX-512 Test..." << std::endl;

    const size_t rows = 64;
    const size_t cols = 128;
    std::vector<float> share_A(rows * cols);
    std::vector<float> share_B(rows * cols);
    std::vector<float> target_W(rows * cols);
    std::vector<float> x(cols);
    std::vector<float> y(rows, 0.0f);
    std::vector<float> y_expected(rows, 0.0f);

    for (size_t i = 0; i < cols; ++i) {
        x[i] = 1.0f + 0.1f * (i % 5);
    }

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t idx = r * cols + c;
            int ternary = (int)(idx % 3) - 1; // signed integer: -1, 0, 1
            float val = static_cast<float>(ternary);
            target_W[idx] = val;
            share_A[idx] = 0.5f * val + 0.25f;
            share_B[idx] = 0.5f * val - 0.25f;
        }
    }

    assert(weft::AzumayaSecretShareEngine::verify_reconstruction(
        share_A.data(), share_B.data(), target_W.data(), rows * cols
    ));

    weft::AzumayaSecretShareEngine::evaluate_2party_gemv(
        share_A.data(), share_B.data(), x.data(), y.data(), rows, cols
    );

    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            acc += target_W[r * cols + c] * x[c];
        }
        y_expected[r] = acc;
        float diff = std::abs(y[r] - y_expected[r]);
        if (diff >= 1e-4f) {
            std::cout << "r=" << r << " y=" << y[r] << " exp=" << y_expected[r] << " diff=" << diff << std::endl;
        }
        assert(diff < 1e-3f);
    }

    std::cout << "[PASS] Azumaya Secret Share Engine verified successfully." << std::endl;
    return 0;
}
