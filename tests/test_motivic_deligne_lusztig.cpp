// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_motivic_deligne_lusztig.h"

int main() {
    std::cout << "[TEST] Running Deligne-Lusztig Parity Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> line1(dim);
    std::vector<float> line2(dim);
    std::vector<float> parity(dim, 0.0f);
    std::vector<float> recovered(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        line1[i] = 1.0f + 0.05f * (i % 7);
        line2[i] = 2.0f - 0.02f * (i % 9);
    }

    weft::DeligneLusztigParityEngine::generate_2line_parity(
        line1.data(), line2.data(), parity.data(), dim
    );

    weft::DeligneLusztigParityEngine::recover_line1(
        parity.data(), line2.data(), recovered.data(), dim
    );

    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(recovered[i] - line1[i]) < 1e-5f);
    }

    std::cout << "[PASS] Deligne-Lusztig Parity Engine verified successfully." << std::endl;
    return 0;
}
