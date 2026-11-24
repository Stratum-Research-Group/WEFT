// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Equivariant Chow Localization Invariants (Paper 217)

#include "csrc/cli/weft_equivariant_chow_routing.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_localize_and_reduce_gradients() {
    const size_t N = 32;
    std::vector<float> local_grads(N, 10.0f);
    std::vector<float> euler_weights(N, 2.0f);
    std::vector<float> global_out(N, 0.0f);

    weft::equivariant_chow::localize_and_reduce_gradients_avx512(
        local_grads.data(), euler_weights.data(), global_out.data(), N
    );

    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(global_out[i] - 5.0f) < 1e-6f);
    }
    std::cout << "[PASS] test_localize_and_reduce_gradients (N=32 verified)" << std::endl;
}

void test_multi_device_equivariant_allreduce() {
    const size_t N = 16;
    std::vector<std::vector<float>> node_grads = {
        std::vector<float>(N, 2.0f),
        std::vector<float>(N, 4.0f),
        std::vector<float>(N, 6.0f)
    };
    std::vector<float> euler_weights(N, 2.0f);
    std::vector<float> reduced(N, 0.0f);

    // Node 1: 2/2 = 1.0
    // Node 2: 4/2 = 2.0
    // Node 3: 6/2 = 3.0
    // Sum = 6.0
    weft::equivariant_chow::multi_device_equivariant_allreduce(
        node_grads, euler_weights, reduced
    );

    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(reduced[i] - 6.0f) < 1e-5f);
    }
    std::cout << "[PASS] test_multi_device_equivariant_allreduce (3 nodes, sum 6.0 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 217: Equivariant Chow Localization Unit Tests ===" << std::endl;
    test_localize_and_reduce_gradients();
    test_multi_device_equivariant_allreduce();
    std::cout << "All Equivariant Chow Localization tests PASSED." << std::endl;
    return 0;
}
