#include <iostream>
#include <cassert>
#include <vector>
#include "csrc/cli/weft_motivic_steenrod_recovery.h"

int main() {
    std::cout << "[Test] Initializing Motivic Steenrod Recovery SIMD Test..." << std::endl;

    const size_t count = 32;
    std::vector<uint32_t> original(count, 0x12345678);
    std::vector<uint32_t> corrupted(count, 0x12345678);
    std::vector<uint32_t> error_masks(count, 0);
    std::vector<uint32_t> out_clean(count, 0);

    // Inject simulated SEU bitflips
    for (size_t i = 0; i < count; ++i) {
        uint32_t mask = (1u << (i % 32));
        error_masks[i] = mask;
        corrupted[i] ^= mask; // corrupt
    }

    weft::avx512_steenrod_syndrome_recovery(
        corrupted.data(),
        error_masks.data(),
        out_clean.data(),
        count
    );

    for (size_t i = 0; i < count; ++i) {
        assert(out_clean[i] == original[i]);
    }

    std::cout << "[Test] Clean Word [0]: 0x" << std::hex << out_clean[0] << std::dec << std::endl;
    std::cout << "[Test] Motivic Steenrod Recovery SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
