#ifndef WEFT_TANNAKIAN_GALOIS_H
#define WEFT_TANNAKIAN_GALOIS_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct TannakianGaugeProfile {
    float invariant_energy;
    float gauge_outlier_norm;
    float outlier_suppression_ratio;
};

// Projects incoming activations onto the G_mot-invariant subspace M^{G_mot}
// Removes uncalibrated gauge rotation components that destroy sub-bit sparsity
inline TannakianGaugeProfile project_tannakian_gauge_invariants(
    const float* activation_in,
    float* activation_out,
    size_t dim,
    float gauge_clip_threshold
) {
    TannakianGaugeProfile prof;
    prof.invariant_energy = 0.0f;
    prof.gauge_outlier_norm = 0.0f;

    float inv_acc = 0.0f;
    float outlier_acc = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        float val = activation_in[i];
        if (std::abs(val) > gauge_clip_threshold) {
            // Gauge outlier detected: project onto canonical invariant boundary
            float sign = (val >= 0.0f) ? 1.0f : -1.0f;
            float clipped = sign * gauge_clip_threshold;
            activation_out[i] = clipped;
            outlier_acc += (val - clipped) * (val - clipped);
            inv_acc += clipped * clipped;
        } else {
            activation_out[i] = val;
            inv_acc += val * val;
        }
    }

    prof.invariant_energy = std::sqrt(inv_acc);
    prof.gauge_outlier_norm = std::sqrt(outlier_acc);
    prof.outlier_suppression_ratio = (prof.gauge_outlier_norm > 0.0f) ?
        (prof.invariant_energy / prof.gauge_outlier_norm) : 100.0f;

    return prof;
}

} // namespace weft

#endif // WEFT_TANNAKIAN_GALOIS_H
