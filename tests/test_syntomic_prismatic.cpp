// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Syntomic Prismatic Speculative Verification Invariants (Paper 216)

#include "csrc/cli/weft_syntomic_prismatic.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_syntomic_verification() {
    const size_t dim = 32;
    std::vector<float> target(dim, 1.0f);
    std::vector<float> draft_ok(dim, 1.05f); // diff = 0.05 <= 0.10
    std::vector<float> draft_bad(dim, 1.05f);
    draft_bad[15] = 1.30f; // diff = 0.30 > 0.10

    bool ok = weft::syntomic_prismatic::verify_tokens_syntomic_avx512(
        target.data(), draft_ok.data(), dim, 0.10f
    );
    assert(ok == true);

    bool bad = weft::syntomic_prismatic::verify_tokens_syntomic_avx512(
        target.data(), draft_bad.data(), dim, 0.10f
    );
    assert(bad == false);

    std::cout << "[PASS] test_syntomic_verification (acceptance and rejection verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 216: Syntomic Prismatic Unit Tests ===" << std::endl;
    test_syntomic_verification();
    std::cout << "All Syntomic Prismatic tests PASSED." << std::endl;
    return 0;
}
