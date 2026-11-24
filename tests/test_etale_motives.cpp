#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_etale_motives.h"

int main() {
    std::cout << "[TEST] Running Étale Motives & ℓ-Adic Weight Filtration Invariant Test..." << std::endl;

    const size_t DIM = 16;
    // 16 values = 4 bytes of 2-bit values
    std::vector<uint8_t> raw_bits = {0x1B, 0xE4, 0x00, 0x55}; // test patterns
    std::vector<float> dequant_out(DIM, 0.0f);

    float q = 4.0f; // q=4 -> q^(-w/2) = 4^(-1) = 0.25 for w=2
    uint32_t w = 2;

    weft::EtaleDequantProfile prof = weft::dequantize_etale_weight_filtration(
        raw_bits.data(), dequant_out.data(), DIM, q, w
    );

    std::cout << "  Dequantized Pure Weight Norm: " << prof.pure_weight_norm << std::endl;
    std::cout << "  Is Pure: " << (prof.is_pure ? "true" : "false") << std::endl;

    assert(prof.is_pure);
    assert(prof.pure_weight_norm > 0.0f);

    // Verify first value: 0x1B has lowest 2 bits = 0x3 (binary 11) -> signed -1
    // Scale factor = 4^(-1) = 0.25 -> dequant = -0.25
    float expected_first = -0.25f;
    std::cout << "  dequant_out[0] = " << dequant_out[0] << " (expected: " << expected_first << ")" << std::endl;
    assert(std::abs(dequant_out[0] - expected_first) < 1e-5f);

    std::cout << "[PASS] All Étale Motives & ℓ-Adic Weight Filtration invariants verified." << std::endl;
    return 0;
}
