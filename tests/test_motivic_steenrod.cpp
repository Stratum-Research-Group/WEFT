// test_motivic_steenrod.cpp
// Milestone Paper 233: Verification of AVX-512 Motivic Steenrod Operations & Error Correction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_motivic_steenrod.h"
#include <iostream>
#include <cassert>

void test_motivic_syndrome_clean() {
    weft::MotivicTernaryBlock block{0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
    weft::MotivicSyndrome syn = weft::compute_motivic_syndrome(block);

    // Verify repair on uncorrupted block reports success without changing data
    bool ok = weft::repair_motivic_ternary_block(block, syn);
    assert(ok);
    assert(block.mask_pos == 0x0123456789ABCDEFULL);
}

void test_motivic_syndrome_correction() {
    weft::MotivicTernaryBlock block{0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
    weft::MotivicSyndrome original_syn = weft::compute_motivic_syndrome(block);

    // Introduce single bit flip at position 17
    const int flip_pos = 17;
    block.mask_pos ^= (1ULL << flip_pos);
    assert(block.mask_pos != 0x0123456789ABCDEFULL);

    // Repair bit flip
    bool ok = weft::repair_motivic_ternary_block(block, original_syn);
    assert(ok);
    assert(block.mask_pos == 0x0123456789ABCDEFULL);
}

int main() {
    std::cout << "[Paper 233] Running Motivic Steenrod Sub-Bit Error Correction Unit Tests...\n";
    test_motivic_syndrome_clean();
    test_motivic_syndrome_correction();
    std::cout << "[Paper 233] All unit tests PASSED successfully.\n";
    return 0;
}
