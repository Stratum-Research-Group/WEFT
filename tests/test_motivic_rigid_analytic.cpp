// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include "../csrc/cli/weft_motivic_rigid_analytic.h"

int main() {
    std::cout << "[TEST] Running Rigid Analytic Checkpoint Engine AVX-512 Test..." << std::endl;

    alignas(64) uint8_t valid_header[64];
    alignas(64) uint8_t invalid_header[64];
    alignas(64) uint8_t expected_magic[64];

    std::memset(valid_header, 0, 64);
    std::memset(invalid_header, 0, 64);
    std::memset(expected_magic, 0, 64);

    std::memcpy(valid_header, "WEFT0004", 8);
    std::memcpy(invalid_header, "CORRUPT!", 8);
    std::memcpy(expected_magic, "WEFT0004", 8);

    assert(weft::RigidAnalyticCheckpointEngine::validate_header_64b(valid_header, expected_magic));
    assert(!weft::RigidAnalyticCheckpointEngine::validate_header_64b(invalid_header, expected_magic));

    std::cout << "[PASS] Rigid Analytic Checkpoint Engine verified successfully." << std::endl;
    return 0;
}
