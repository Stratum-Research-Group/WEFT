// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_subbit_popcount_gemv.h"

int main() {
    std::cout << "[TEST] Running Sub-Bit Popcount Reducer AVX-512 Test..." << std::endl;

    const size_t num_qwords = 4; // 256 weights
    std::vector<uint64_t> pos_plane(num_qwords, 0);
    std::vector<uint64_t> neg_plane(num_qwords, 0);

    // Set 10 positive bits and 3 negative bits
    pos_plane[0] = 0x3FF; // 10 ones
    neg_plane[0] = 0x7;   // 3 ones

    int32_t dot = weft::SubbitPopcountReducer::compute_popcount_ternary_dot(
        pos_plane.data(), neg_plane.data(), num_qwords
    );

    assert(dot == (10 - 3)); // 7
    std::cout << "[PASS] Sub-Bit Popcount Reducer verified successfully (dot: " << dot << ")." << std::endl;
    return 0;
}
