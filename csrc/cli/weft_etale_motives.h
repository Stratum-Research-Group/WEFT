#ifndef WEFT_ETALE_MOTIVES_H
#define WEFT_ETALE_MOTIVES_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct EtaleDequantProfile {
    float pure_weight_norm;
    float graded_residual;
    bool is_pure;
};

// Dequantizes multi-stratum sub-bit weights via Deligne l-adic weight filtration normalization
// Gr_i^W W = (W_i - W_{i-1}) * q^(-i/2)
inline EtaleDequantProfile dequantize_etale_weight_filtration(
    const uint8_t* raw_bits,
    float* dequant_out,
    size_t dim,
    float frobenius_q,
    uint32_t weight_w
) {
    EtaleDequantProfile prof;
    prof.pure_weight_norm = 0.0f;
    prof.graded_residual = 0.0f;

    // Frobenius scaling factor q^(-w/2)
    float q_factor = std::pow(frobenius_q, -0.5f * static_cast<float>(weight_w));
    float norm_acc = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        // Extract 2-bit signed quantized value: values in {-2, -1, 0, 1}
        uint8_t byte_val = raw_bits[i / 4];
        int shift = (i % 4) * 2;
        int raw_val = (byte_val >> shift) & 0x03;
        int signed_val = (raw_val >= 2) ? (raw_val - 4) : raw_val;

        float dequant = static_cast<float>(signed_val) * q_factor;
        dequant_out[i] = dequant;
        norm_acc += dequant * dequant;
    }

    prof.pure_weight_norm = std::sqrt(norm_acc);
    prof.is_pure = (prof.pure_weight_norm >= 0.0f);
    return prof;
}

} // namespace weft

#endif // WEFT_ETALE_MOTIVES_H
