#include "../csrc/kernels/avx512_subbit_engine.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Test AVX-512 Sub-Bit Ternary Dot Product Engine ===" << std::endl;

    size_t n = 64;
    std::vector<int8_t> act(n, 2);
    // uint64_t mask: 32 ones (+1), 32 zeros (-1)
    uint64_t mask = 0xFFFFFFFF00000000ULL;
    // Expected dot product: 32 * (2 * 1) + 32 * (2 * -1) = 64 - 64 = 0
    int32_t dot0 = weft::AVX512SubBitEngine::dot_product_binary(act.data(), &mask, n);
    assert(dot0 == 0);

    // All ones mask: 64 * (2 * 1) = 128
    uint64_t mask_all1 = 0xFFFFFFFFFFFFFFFFULL;
    int32_t dot1 = weft::AVX512SubBitEngine::dot_product_binary(act.data(), &mask_all1, n);
    assert(dot1 == 128);

    // All zeros mask: 64 * (2 * -1) = -128
    uint64_t mask_all0 = 0x0ULL;
    int32_t dot_neg = weft::AVX512SubBitEngine::dot_product_binary(act.data(), &mask_all0, n);
    assert(dot_neg == -128);

    std::cout << "[PASS] AVX-512 Sub-Bit Ternary Dot Product Verified: dot0=" << dot0 << " dot1=" << dot1 << " dot_neg=" << dot_neg << std::endl;
    return 0;
}
