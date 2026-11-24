// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Motivic Infinite-Loop Spaces & Algebraic K-Theory Invariants (Paper 218)

#include "csrc/cli/weft_motivic_loop_spaces.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_activation_stack_lifecycle() {
    const size_t max_layers = 4;
    const size_t dim = 32;
    weft::motivic_loop_spaces::ActivationStackAVX512 stack(max_layers, dim);

    assert(stack.depth() == 0);
    assert(stack.capacity() == 4);

    std::vector<float> l1(dim, 1.0f);
    std::vector<float> l2(dim, 2.0f);
    std::vector<float> out(dim, 0.0f);

    assert(stack.push(l1.data()) == true);
    assert(stack.depth() == 1);
    assert(stack.push(l2.data()) == true);
    assert(stack.depth() == 2);

    // Pop l2
    assert(stack.pop(out.data()) == true);
    assert(stack.depth() == 1);
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(out[i] - 2.0f) < 1e-6f);
    }

    // Pop l1
    assert(stack.pop(out.data()) == true);
    assert(stack.depth() == 0);
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(out[i] - 1.0f) < 1e-6f);
    }

    // Underflow check
    assert(stack.pop(out.data()) == false);

    std::cout << "[PASS] test_activation_stack_lifecycle (push, pop, underflow verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 218: Motivic Infinite-Loop Spaces Unit Tests ===" << std::endl;
    test_activation_stack_lifecycle();
    std::cout << "All Motivic Infinite-Loop Spaces tests PASSED." << std::endl;
    return 0;
}
