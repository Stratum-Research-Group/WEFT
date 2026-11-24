#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_fontaine_attestation.h"

int main() {
    std::cout << "[TEST] Running Fontaine Period Ring Cryptographic Attestation Test..." << std::endl;

    // Test 1: Hodge jump evaluation
    // 1 -> clz = 31
    // 2 -> clz = 30
    // 0 -> 32
    uint32_t j1 = weft::compute_fontaine_hodge_jump(1);
    uint32_t j2 = weft::compute_fontaine_hodge_jump(2);
    uint32_t j0 = weft::compute_fontaine_hodge_jump(0);

    std::cout << "  Hodge jumps: j(1)=" << j1 << ", j(2)=" << j2 << ", j(0)=" << j0 << std::endl;
    assert(j1 == 31);
    assert(j2 == 30);
    assert(j0 == 32);

    // Test 2: In-register attestation verification
    const size_t DIM = 16;
    std::vector<uint32_t> weights(DIM);
    std::vector<uint32_t> expected_jumps(DIM);

    for (size_t i = 0; i < DIM; ++i) {
        weights[i] = 1u << (i % 16);
        expected_jumps[i] = weft::compute_fontaine_hodge_jump(weights[i]);
    }

    bool is_valid = weft::verify_weight_fontaine_attestation_simd(weights.data(), expected_jumps.data(), DIM);
    std::cout << "  Untampered weights verification: " << is_valid << std::endl;
    assert(is_valid == true);

    // Simulate tampering on higher bit altering leading zeros
    weights[3] = 0; // Alters clz from 28 to 32
    bool is_tampered = weft::verify_weight_fontaine_attestation_simd(weights.data(), expected_jumps.data(), DIM);
    std::cout << "  Tampered weights verification: " << is_tampered << " (Expected: false)" << std::endl;
    assert(is_tampered == false);

    std::cout << "[PASS] All Fontaine Period Ring Attestation invariants verified." << std::endl;
    return 0;
}
