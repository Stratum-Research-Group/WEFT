// test_berkovich_eviction.cpp
// Milestone Paper 235: Verification of AVX-512 Berkovich Skeletal KV Eviction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_berkovich_eviction.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_berkovich_eviction_kernel() {
    const size_t N = 32;
    std::vector<float> core_dist(N, 0.0f);
    std::vector<uint32_t> is_anchor(N, 0);
    std::vector<uint8_t> evict_flags(N, 0);

    // Set first 8 tokens as system prompt anchors with large distance
    for (size_t i = 0; i < 8; ++i) {
        core_dist[i] = 100.0f;
        is_anchor[i] = 1;
    }

    // Set tokens 8-15 as distant conversational leaves
    for (size_t i = 8; i < 16; ++i) {
        core_dist[i] = 50.0f;
        is_anchor[i] = 0;
    }

    // Set tokens 16-31 as recent tokens near core
    for (size_t i = 16; i < 32; ++i) {
        core_dist[i] = 5.0f;
        is_anchor[i] = 0;
    }

    float tau = 20.0f;
    weft::avx512_berkovich_evict_mask(core_dist.data(), is_anchor.data(), evict_flags.data(), N, tau);

    // Invariant 1: Anchors are never evicted
    for (size_t i = 0; i < 8; ++i) {
        assert(evict_flags[i] == 0);
    }

    // Invariant 2: Distant non-anchors (dist 50 > 20) are evicted
    for (size_t i = 8; i < 16; ++i) {
        assert(evict_flags[i] == 1);
    }

    // Invariant 3: Near non-anchors (dist 5 <= 20) are kept
    for (size_t i = 16; i < 32; ++i) {
        assert(evict_flags[i] == 0);
    }
}

int main() {
    std::cout << "[Paper 235] Running Berkovich Skeletal KV Eviction Unit Tests...\n";
    test_berkovich_eviction_kernel();
    std::cout << "[Paper 235] All unit tests PASSED successfully.\n";
    return 0;
}
