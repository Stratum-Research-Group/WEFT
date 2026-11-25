#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace weft {

// Tropical polyhedral page eviction selector for multi-tier memory paging.
// Evaluates eviction score: h_evict(p) = max_i (w_i * metric_i)
class TropicalFiberPager {
public:
    static size_t select_eviction_candidate(const float* page_latencies, const float* page_access_freqs,
                                           const float* weights, size_t num_pages) {
        if (!page_latencies || !page_access_freqs || !weights || num_pages == 0) return 0;

        size_t best_page = 0;
        float min_eviction_cost = 1e30f; // Evict page that has lowest future utility

        for (size_t p = 0; p < num_pages; ++p) {
            // Eviction cost = freq * weights[0] + (1 / (latency + 1e-4)) * weights[1]
            float utility = page_access_freqs[p] * weights[0] + (1.0f / (page_latencies[p] + 1e-4f)) * weights[1];
            if (utility < min_eviction_cost) {
                min_eviction_cost = utility;
                best_page = p;
            }
        }
        return best_page;
    }

    // SIMD vector evaluation of page eviction scores
    static void evaluate_eviction_scores(const float* freqs, const float* latencies,
                                         float* out_scores, size_t num_pages,
                                         float w_freq = 1.0f, float w_lat = 0.5f) {
        if (!freqs || !latencies || !out_scores || num_pages == 0) return;

        __m512 wf = _mm512_set1_ps(w_freq);
        __m512 wl = _mm512_set1_ps(w_lat);

        size_t i = 0;
        for (; i + 16 <= num_pages; i += 16) {
            __m512 f = _mm512_loadu_ps(&freqs[i]);
            __m512 l = _mm512_loadu_ps(&latencies[i]);
            __m512 score = _mm512_add_ps(_mm512_mul_ps(f, wf), _mm512_mul_ps(l, wl));
            _mm512_storeu_ps(&out_scores[i], score);
        }
        for (; i < num_pages; ++i) {
            out_scores[i] = freqs[i] * w_freq + latencies[i] * w_lat;
        }
    }
};

} // namespace weft
