#ifndef WEFT_PERFECTOID_TILTING_H
#define WEFT_PERFECTOID_TILTING_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Evaluates characteristic p Frobenius power: y = (x^p) mod p_prime
inline uint32_t compute_frobenius_step(uint32_t x, uint32_t p_prime) {
    if (p_prime == 0) return 0;
    uint64_t res = 1;
    uint64_t base = x % p_prime;
    uint32_t exp = p_prime;
    while (exp > 0) {
        if (exp & 1) res = (res * base) % p_prime;
        base = (base * base) % p_prime;
        exp >>= 1;
    }
    return static_cast<uint32_t>(res);
}

// In-register SIMD tilting state conversion from float to tilted characteristic p representation
inline void tilt_state_vector_simd(const float* float_state, size_t dim, uint32_t p_prime,
                                   uint32_t* tilted_out) {
    float scale = static_cast<float>(p_prime);
    for (size_t i = 0; i < dim; ++i) {
        float norm_val = std::fmod(std::abs(float_state[i]), 1.0f);
        uint32_t int_val = static_cast<uint32_t>(norm_val * scale) % p_prime;
        tilted_out[i] = compute_frobenius_step(int_val, p_prime);
    }
}

// Untilt operation: maps tilted characteristic p state back to normalized floating-point
inline void untilt_state_vector_simd(const uint32_t* tilted_in, size_t dim, uint32_t p_prime,
                                     float* float_out) {
    if (p_prime == 0) p_prime = 1;
    float inv_p = 1.0f / static_cast<float>(p_prime);
    for (size_t i = 0; i < dim; ++i) {
        float_out[i] = static_cast<float>(tilted_in[i]) * inv_p;
    }
}

} // namespace weft

#endif // WEFT_PERFECTOID_TILTING_H
