#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_berkovich_spaces.h"

int main() {
    std::cout << "[TEST] Running Berkovich Space Token Hierarchy Invariant Test..." << std::endl;

    // Test 1: 2-adic valuation: diff=8 (2^3) -> valuation 3
    uint32_t v1 = weft::compute_2adic_valuation(16, 24);
    std::cout << "  Valuation v2(|16 - 24| = 8): " << v1 << " (Expected: 3)" << std::endl;
    assert(v1 == 3);

    // Test 2: In-register tree search
    std::vector<uint32_t> keys = {0, 4, 8, 12, 16, 20, 24, 28}; // All multiples of 4
    std::vector<uint32_t> matches(keys.size());

    // Query 0 with min_valuation = 3 (multiples of 8) -> 0, 8, 16, 24 (4 matches)
    size_t count = weft::query_berkovich_tree_simd(keys.data(), keys.size(), 0, 3, matches.data());
    std::cout << "  Retained skeletal tokens (v2 >= 3): " << count << " / " << keys.size() << std::endl;
    assert(count == 4);

    std::cout << "[PASS] All Berkovich Space Token Hierarchy invariants verified." << std::endl;
    return 0;
}
