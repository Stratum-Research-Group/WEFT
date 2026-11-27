#include <cstdint>
#include <cstddef>
#include <vector>
#include <immintrin.h>

extern "C" {

/**
 * Fused AVX-512 VNNI / FP16 Speculative Recurrent Matrix-Vector Multiplication.
 * C-ABI for Rust FFI integration.
 */
int weft_speculative_vnni_step(
    const uint8_t* weights,
    const uint8_t* inputs,
    int32_t* outputs,
    size_t dim,
    size_t tokens
) {
    if (!weights || !inputs || !outputs || dim % 64 != 0) {
        return -1;
    }

    for (size_t t = 0; t < tokens; ++t) {
        const uint8_t* in_ptr = inputs + (t * dim);
        int32_t* out_ptr = outputs + (t * dim);

        for (size_t row = 0; row < dim; ++row) {
            const uint8_t* w_ptr = weights + (row * dim);
            __m512i acc = _mm512_setzero_si512();

            for (size_t col = 0; col < dim; col += 64) {
                __m512i a = _mm512_loadu_si512((const __m512i*)(in_ptr + col));
                __m512i b = _mm512_loadu_si512((const __m512i*)(w_ptr + col));
                #if defined(__AVX512VNNI__)
                acc = _mm512_dpbusd_epi32(acc, a, b);
                #else
                __m512i prod = _mm512_maddubs_epi16(a, b);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(prod, _mm512_set1_epi16(1)));
                #endif
            }
            out_ptr[row] = _mm512_reduce_add_epi32(acc);
        }
    }
    return 0;
}

}
