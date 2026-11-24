// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Toric Arithmetic Chow Attestation Invariants (Paper 215)

#include "csrc/cli/weft_toric_arithmetic_chow.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_arithmetic_degree_properties() {
    int64_t deg = weft::toric_attestation::compute_arithmetic_degree(100, 30);
    assert(deg == 70);

    int64_t zero_deg = weft::toric_attestation::compute_arithmetic_degree(0, 0);
    assert(zero_deg == 0);
    std::cout << "[PASS] test_arithmetic_degree_properties" << std::endl;
}

void test_weight_attestation_avx512() {
    const size_t N = 64;
    std::vector<uint32_t> weights(N, 0x0F0F0F0F); // 16 set bits per word
    int64_t expected = 16 * N;

    int64_t obs = weft::toric_attestation::aggregate_attestation_avx512(weights.data(), N);
    assert(obs == expected);

    bool ok = weft::toric_attestation::verify_weight_attestation(weights.data(), N, expected);
    assert(ok == true);

    // Tamper single bit
    weights[0] ^= 0x1;
    bool tampered_ok = weft::toric_attestation::verify_weight_attestation(weights.data(), N, expected);
    assert(tampered_ok == false);

    std::cout << "[PASS] test_weight_attestation_avx512 (tamper detection verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 215: Toric Arithmetic Chow Attestation Unit Tests ===" << std::endl;
    test_arithmetic_degree_properties();
    test_weight_attestation_avx512();
    std::cout << "All Toric Arithmetic Chow Attestation tests PASSED." << std::endl;
    return 0;
}
