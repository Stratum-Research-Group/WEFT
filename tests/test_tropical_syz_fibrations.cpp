// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_tropical_syz_fibrations.h"

int main() {
    std::cout << "[TEST] Running SYZ Shard Accumulator AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> shard1(dim);
    std::vector<float> shard2(dim);
    std::vector<float> y_out(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        shard1[i] = 1.0f + 0.1f * (i % 5);
        shard2[i] = 2.0f - 0.05f * (i % 6);
    }

    weft::SYZShardAccumulator::accumulate_shard(shard1.data(), 0.5f, y_out.data(), dim);
    weft::SYZShardAccumulator::accumulate_shard(shard2.data(), 0.25f, y_out.data(), dim);

    for (size_t i = 0; i < dim; ++i) {
        float expected = 0.5f * shard1[i] + 0.25f * shard2[i];
        assert(std::abs(y_out[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] SYZ Shard Accumulator verified successfully." << std::endl;
    return 0;
}
