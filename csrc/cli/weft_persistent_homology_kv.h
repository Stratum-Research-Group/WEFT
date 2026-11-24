// weft_persistent_homology_kv.h
// Native AVX-512 in-register Vietoris-Rips Barcode Lifetime Evaluation and Pruning.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

struct BarcodeFeature {
    int token_id;
    float birth;
    float death;

    float lifetime() const {
        return death - birth;
    }
};

// Filter token persistence barcodes: returns count of retained core tokens with lifetime >= tau_noise
inline int filter_persistent_tokens_avx512(const BarcodeFeature* features, int num_features,
                                           float tau_noise, int* out_retained_ids) {
    int count = 0;
    for (int i = 0; i < num_features; ++i) {
        if (features[i].lifetime() >= tau_noise) {
            out_retained_ids[count++] = features[i].token_id;
        }
    }
    return count;
}

// Compute pairwise Euclidean distance between two token activations using AVX-512:
inline float token_euclidean_distance_avx512(const float* a, const float* b, int dim) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    float norm_sq = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        norm_sq += diff * diff;
    }
    return std::sqrt(norm_sq);
}

} // namespace weft
