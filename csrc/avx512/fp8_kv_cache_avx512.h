#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace weft {

/**
 * FP8 E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7)
 * Dynamic quantization & dequantization utilities for Ultra-Long KV-Cache.
 */

inline uint8_t float_to_fp8_e4m3(float x, float scale) {
    if (scale <= 1e-12f) return 0;
    float scaled = x / scale;
    if (std::isnan(scaled)) return 0x7F;

    uint32_t b;
    std::memcpy(&b, &scaled, sizeof(float));
    uint32_t sign = (b >> 31) & 0x1;
    int32_t exp = ((b >> 23) & 0xFF) - 127 + 7; // rebias to 7
    uint32_t mant = (b >> 20) & 0x7;

    if (exp <= 0) {
        return static_cast<uint8_t>(sign << 7);
    }
    if (exp >= 15) {
        return static_cast<uint8_t>((sign << 7) | (0xF << 3) | 0x6); // max finite
    }

    return static_cast<uint8_t>((sign << 7) | ((exp & 0xF) << 3) | mant);
}

inline float fp8_e4m3_to_float(uint8_t val, float scale) {
    uint32_t sign = (val >> 7) & 0x1;
    uint32_t exp = (val >> 3) & 0xF;
    uint32_t mant = val & 0x7;

    if (exp == 0 && mant == 0) return 0.0f;

    float f_exp = std::pow(2.0f, static_cast<float>(exp) - 7.0f);
    float f_mant = 1.0f + static_cast<float>(mant) / 8.0f;
    float res = (sign ? -1.0f : 1.0f) * f_exp * f_mant;

    return res * scale;
}

#if defined(__AVX512F__)
inline void dequantize_fp8_e4m3_block_avx512(
    const uint8_t* fp8_ptr,
    float* out_ptr,
    float scale,
    int64_t count
) {
    for (int64_t i = 0; i < count; ++i) {
        out_ptr[i] = fp8_e4m3_to_float(fp8_ptr[i], scale);
    }
}
#endif

} // namespace weft
