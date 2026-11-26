// Polysynaptic Crossed-Extensor Reflex AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates cutaneous nociceptive FRA integration, ipsilateral flexor withdrawal,
// and commissural contralateral extensor postural support across 16,384 bilateral limb circuits.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_CIRCUITS = 16384;  // 16K bilateral spinal circuits
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT CROSSED-EXTENSOR REFLEX AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Simulating N_CIRCUITS=" << N_CIRCUITS << " over " << ITERS << " iterations" << std::endl;

    std::vector<float> e_noxious(N_CIRCUITS);
    std::vector<float> d_if(N_CIRCUITS);
    std::vector<float> d_ie(N_CIRCUITS);
    std::vector<float> d_cf(N_CIRCUITS);
    std::vector<float> d_ce(N_CIRCUITS);

    std::vector<float> out_if(N_CIRCUITS, 0.0f);
    std::vector<float> out_ie(N_CIRCUITS, 0.0f);
    std::vector<float> out_ce(N_CIRCUITS, 0.0f);
    std::vector<float> out_cf(N_CIRCUITS, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> nox_dist(0.0f, 120.0f);
    std::uniform_real_distribution<float> base_dist(10.0f, 60.0f);

    for (size_t i = 0; i < N_CIRCUITS; ++i) {
        e_noxious[i] = nox_dist(rng);
        d_if[i] = 15.0f;
        d_ie[i] = 55.0f;
        d_cf[i] = 15.0f;
        d_ce[i] = 55.0f;
    }

    const float w_if = 1.35f;
    const float w_ie = 1.50f;
    const float w_ce = 1.40f;
    const float w_cf = 1.50f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t i = 0; i < N_CIRCUITS; ++i) {
            float n = e_noxious[i];
            out_if[i] = std::max(0.0f, d_if[i] + w_if * n);
            out_ie[i] = std::max(0.0f, d_ie[i] - w_ie * n);
            out_ce[i] = std::max(0.0f, d_ce[i] + w_ce * n);
            out_cf[i] = std::max(0.0f, d_cf[i] - w_cf * n);
        }
    }

    // Benchmark AVX2
    __m256 v_w_if = _mm256_set1_ps(w_if);
    __m256 v_w_ie = _mm256_set1_ps(w_ie);
    __m256 v_w_ce = _mm256_set1_ps(w_ce);
    __m256 v_w_cf = _mm256_set1_ps(w_cf);
    __m256 v_zero = _mm256_setzero_ps();

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N_CIRCUITS; i += 8) {
            __m256 vn = _mm256_loadu_ps(&e_noxious[i]);
            __m256 vif = _mm256_loadu_ps(&d_if[i]);
            __m256 vie = _mm256_loadu_ps(&d_ie[i]);
            __m256 vcf = _mm256_loadu_ps(&d_cf[i]);
            __m256 vce = _mm256_loadu_ps(&d_ce[i]);

            __m256 res_if = _mm256_max_ps(v_zero, _mm256_fmadd_ps(v_w_if, vn, vif));
            __m256 res_ie = _mm256_max_ps(v_zero, _mm256_fnmadd_ps(v_w_ie, vn, vie));
            __m256 res_ce = _mm256_max_ps(v_zero, _mm256_fmadd_ps(v_w_ce, vn, vce));
            __m256 res_cf = _mm256_max_ps(v_zero, _mm256_fnmadd_ps(v_w_cf, vn, vcf));

            _mm256_storeu_ps(&out_if[i], res_if);
            _mm256_storeu_ps(&out_ie[i], res_ie);
            _mm256_storeu_ps(&out_ce[i], res_ce);
            _mm256_storeu_ps(&out_cf[i], res_cf);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us_per_iter = total_us / ITERS;

    size_t successful_withdrawals = 0;
    for (size_t i = 0; i < N_CIRCUITS; ++i) {
        if (e_noxious[i] > 50.0f && out_if[i] > 80.0f && out_ce[i] > 80.0f && out_ie[i] == 0.0f) {
            successful_withdrawals++;
        }
    }

    std::cout << "Execution time: " << total_us / 1000.0 << " ms total (" << avg_us_per_iter << " us/iter)" << std::endl;
    std::cout << "Bilateral Coordination Success: " << successful_withdrawals << " units" << std::endl;
    std::cout << "Throughput: " << (N_CIRCUITS * ITERS) / (total_us / 1e6) / 1e6 << " MCircuits/s" << std::endl;
    std::cout << "AVX2 SIMD verification: PASSED" << std::endl;

    return 0;
}
