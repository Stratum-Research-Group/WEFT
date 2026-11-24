// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Derived Azumaya Speculative Rollback Invariants (Paper 230)

#include "csrc/cli/weft_derived_azumaya.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_transactional_rollback() {
    const size_t capacity = 16;
    const size_t dim = 32;
    weft::derived_azumaya::TransactionalKVCache cache(capacity, dim);

    std::vector<float> token(dim, 5.0f);

    // Write 4 tokens
    for (int i = 0; i < 4; ++i) cache.append_token(token.data());
    assert(cache.current_pos() == 4);

    // Set checkpoint at position 4
    cache.set_checkpoint();
    assert(cache.checkpoint_pos() == 4);

    // Speculatively draft 3 tokens (pos -> 7)
    for (int i = 0; i < 3; ++i) cache.append_token(token.data());
    assert(cache.current_pos() == 7);

    // Rollback to checkpoint
    cache.rollback_to_checkpoint();
    assert(cache.current_pos() == 4);

    std::cout << "[PASS] test_transactional_rollback (pos 7 -> 4 rollback verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 230: Derived Azumaya Rollback Unit Tests ===" << std::endl;
    test_transactional_rollback();
    std::cout << "All Derived Azumaya Rollback tests PASSED." << std::endl;
    return 0;
}
