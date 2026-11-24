#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_subbit_pack_engine.h"

int main() {
    std::cout << "[Test] Initializing Sub-Bit Pack/Unpack Engine SIMD Test..." << std::endl;

    const size_t count = 32;
    std::vector<float> original(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        if (i % 3 == 0) original[i] = 1.0f;
        else if (i % 3 == 1) original[i] = -1.0f;
        else original[i] = 0.0f;
    }

    std::vector<uint8_t> packed(count / 4, 0);
    weft::avx512_pack_ternary_subbit(original.data(), packed.data(), count);

    std::vector<float> unpacked(count, 999.0f);
    weft::avx512_unpack_ternary_subbit(packed.data(), unpacked.data(), count);

    for (size_t i = 0; i < count; ++i) {
        assert(std::fabs(original[i] - unpacked[i]) < 1e-5f);
    }

    std::cout << "[Test] 32 ternary values packed into " << packed.size() << " bytes (2 bits/weight = 0.25 bytes/weight)." << std::endl;
    std::cout << "[Test] Sub-Bit Pack/Unpack Engine SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
