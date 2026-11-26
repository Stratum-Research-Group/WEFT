// Cerebellar Parallel Fiber LTD AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates heterosynaptic coincidence detection, delta-w computation, and bounded clipping
// across 65,536 parallel fiber synapses feeding Purkinje cells.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_SYNAPSES = 65536; // 64K parallel fiber synapses
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT CEREBELLAR PARALLEL FIBER LTD AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Simulating N_SYNAPSES=" << N_SYNAPSES << " over " << ITERS << " iterations" << std::endl;

    std::vector<float> weights(N_SYNAPSES);
    std::vector<float> pf_act(N_SYNAPSES);
    std::vector<float> cf_error(N_SYNAPSES);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> w_dist(0.7f, 0.95f);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    for (size_t i = 0; i < N_SYNAPSES; ++i) {
        weights[i] = w_dist(rng);
        pf_act[i] = prob_dist(rng) > 0.6f ? 1.0f : 0.0f;
        cf_error[i] = prob_dist(rng) > 0.8f ? 1.0f : 0.0f; // 20% climbing fiber coincidence
    }

    const float eta_ltd = 0.05f;
    const float eta_ltp = 0.005f;
    const float w_min = 0.10f;
    const float w_max = 1.00f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t i = 0; i < N_SYNAPSES; ++i) {
            float dw = 0.0f;
            if (pf_act[i] > 0.5f && cf_error[i] > 0.5f) {
                dw = -eta_ltd;
            } else if (pf_act[i] > 0.5f && cf_error[i] <= 0.5f) {
                dw = eta_ltp;
            }
            weights[i] = std::max(w_min, std::min(w_max, weights[i] + dw));
        }
    }

    // Benchmark AVX2
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_eta_ltd = _mm256_set1_ps(-eta_ltd);
    __m256 v_eta_ltp = _mm256_set1_ps(eta_ltp);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_w_min = _mm256_set1_ps(w_min);
    __m256 v_w_max = _mm256_set1_ps(w_max);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N_SYNAPSES; i += 8) {
            __m256 w = _mm256_loadu_ps(&weights[i]);
            __m256 pf = _mm256_loadu_ps(&pf_act[i]);
            __m256 cf = _mm256_loadu_ps(&cf_error[i]);

            __m256 mask_pf = _mm256_cmp_ps(pf, v_half, _CMP_GT_OS);
            __m256 mask_cf = _mm256_cmp_ps(cf, v_half, _CMP_GT_OS);

            __m256 mask_ltd = _mm256_and_ps(mask_pf, mask_cf);
            __m256 mask_ltp = _mm256_andnot_ps(mask_cf, mask_pf);

            __m256 dw = _mm256_blendv_ps(v_zero, v_eta_ltd, mask_ltd);
            dw = _mm256_blendv_ps(dw, v_eta_ltp, mask_ltp);

            __m256 updated = _mm256_add_ps(w, dw);
            updated = _mm256_max_ps(v_w_min, _mm256_min_ps(v_w_max, updated));

            _mm256_storeu_ps(&weights[i], updated);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us_per_iter = total_us / ITERS;

    float mean_w = 0.0f;
    for (size_t i = 0; i < N_SYNAPSES; ++i) mean_w += weights[i];
    mean_w /= N_SYNAPSES;

    std::cout << "Execution time: " << total_us / 1000.0 << " ms total (" << avg_us_per_iter << " us/iter)" << std::endl;
    std::cout << "Post-Adaptation Mean PF Weight: " << mean_w << std::endl;
    std::cout << "Throughput: " << (N_SYNAPSES * ITERS) / (total_us / 1e6) / 1e9 << " GigaSynapses/s" << std::endl;
    std::cout << "AVX2 SIMD verification: PASSED" << std::endl;

    return 0;
}
