#pragma once
// weft_tropical_berkovich_kv_compaction.h
// Milestone Paper 259: Asymmetric Tropical Berkovich Spectra & Non-Archimedean Amoebas
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Evaluates KV page compaction urgency:
// urgency[i] = max(0, threshold - occupancy[i])
inline void avx512_berkovich_compaction_urgency(
    const float* __restrict__ occupancy,
    float* __restrict__ out_urgency,
    size_t num_pages,
    float threshold
) {
    __m512 v_thresh = _mm512_set1_ps(threshold);
    __m512 v_zero = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= num_pages; i += 16) {
        __m512 v_occ = _mm512_loadu_ps(occupancy + i);
        __m512 v_urg = _mm512_sub_ps(v_thresh, v_occ);
        v_urg = _mm512_max_ps(v_urg, v_zero);
        _mm512_storeu_ps(out_urgency + i, v_urg);
    }

    for (; i < num_pages; ++i) {
        float urg = threshold - occupancy[i];
        out_urgency[i] = (urg > 0.0f) ? urg : 0.0f;
    }
}

} // namespace weft
