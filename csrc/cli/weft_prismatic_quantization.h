#ifndef WEFT_PRISMATIC_QUANTIZATION_H
#define WEFT_PRISMATIC_QUANTIZATION_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Evaluates the Eisenstein polynomial E(u) = u^e + p
inline uint32_t eval_eisenstein_poly(uint32_t u, uint32_t e, uint32_t p) {
    uint32_t u_pow = 1;
    for (uint32_t i = 0; i < e; ++i) {
        u_pow *= u;
    }
    return u_pow + p;
}

// In-register SIMD evaluation of Nygaard filtration for Breuil-Kisin prismatic quantization
inline size_t filter_nygaard_prismatic_weights_simd(const uint32_t* weights_in, size_t dim,
                                                    uint32_t u, uint32_t e, uint32_t p,
                                                    uint32_t* weights_out) {
    uint32_t e_val = eval_eisenstein_poly(u, e, p);
    if (e_val == 0) e_val = 1;

    size_t out_count = 0;
    for (size_t i = 0; i < dim; ++i) {
        uint32_t w = weights_in[i];
        uint32_t phi_w = w * w; // Frobenius lift (p=2)
        if (phi_w % e_val == 0) {
            weights_out[out_count++] = w;
        }
    }
    return out_count;
}

} // namespace weft

#endif // WEFT_PRISMATIC_QUANTIZATION_H
