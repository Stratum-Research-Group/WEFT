// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include "../csrc/cli/weft_motivic_k3_surfaces.h"

int main() {
    std::cout << "[TEST] Running K3 Parity Reconstruction Engine AVX-512 Test..." << std::endl;

    const size_t num_qwords = 8; // 64 bytes
    std::vector<uint64_t> block(num_qwords);

    for (size_t i = 0; i < num_qwords; ++i) {
        block[i] = 0xAAAAAAAAAAAAAAAAULL ^ (i * 0x1111111111111111ULL);
    }

    uint64_t syn = weft::K3ParityReconstructionEngine::compute_block_xor_syndrome(block.data(), num_qwords);
    assert(weft::K3ParityReconstructionEngine::verify_syndrome(block.data(), num_qwords, syn));

    // Inject 1-bit flip
    block[3] ^= 0x1ULL;
    assert(!weft::K3ParityReconstructionEngine::verify_syndrome(block.data(), num_qwords, syn));

    // Correct bit-flip
    block[3] ^= 0x1ULL;
    assert(weft::K3ParityReconstructionEngine::verify_syndrome(block.data(), num_qwords, syn));

    std::cout << "[PASS] K3 Parity Reconstruction Engine verified successfully." << std::endl;
    return 0;
}
