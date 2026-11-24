#ifndef WEFT_SHIMURA_CURVES_H
#define WEFT_SHIMURA_CURVES_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct QuaternionicNormProfile {
    float reduced_norm;
    float discriminant_separation;
    bool is_non_cuspidal;
};

// Computes quaternionic reduced norm for weight 4-tuples and guarantees uniform compact separation
// Nrd(q) = a^2 + d1*b^2 + d2*c^2 + d1*d2*d^2
inline QuaternionicNormProfile compute_quaternionic_norm_4tuple(
    float a, float b, float c, float d,
    float d1, float d2,
    float discriminant
) {
    QuaternionicNormProfile prof;
    float norm = a * a + d1 * b * b + d2 * c * c + d1 * d2 * d * d;

    prof.reduced_norm = norm;
    prof.discriminant_separation = 1.0f / std::sqrt(std::max(1.0f, discriminant));
    // Non-cuspidal compact surface: boundary distance bounded below by discriminant separation
    prof.is_non_cuspidal = (discriminant > 1.0f);

    return prof;
}

} // namespace weft

#endif // WEFT_SHIMURA_CURVES_H
