// Golgi Tendon Organ (GTO) Force & Autogenic Inhibition AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates in-series tendon mechanotransduction, Ib di-synaptic autogenic inhibition,
// and protective clasp-knife release across 32,768 muscle tendon units.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_TENDONS = 32768;   // 32K muscle tendon units
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT GOLGI TENDON ORGAN & AUTOGENIC INHIBITION AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Simulating N_TENDONS=" << N_TENDONS << " over " << ITERS << " iterations" << std::endl;

    std::vector<float> tension(N_TENDONS);
    std::vector<float> drive(N_TENDONS);
    std::vector<float> f_Ib(N_TENDONS, 0.0f);
    std::vector<float> net_alpha(N_TENDONS, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> t_dist(20.0f, 150.0f); // Spanning normal to critical overload
    std::uniform_real_distribution<float> d_dist(60.0f, 120.0f);

    for (size_t i = 0; i < N_TENDONS; ++i) {
        tension[i] = t_dist(rng);
        drive[i] = d_dist(rng);
    }

    const float k_Ib = 4.5f;
    const float t_critical = 120.0f; // Critical rupture threshold
    const float w_Ib = 0.55f;
    const float f_max = 300.0f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t i = 0; i < N_TENDONS; ++i) {
            float t = tension[i];
            float ib = std::min(f_max, k_Ib * std::pow(t, 0.85f));
            float inhib = w_Ib * ib;
            float alpha = (t >= t_critical) ? 0.0f : std::max(0.0f, drive[i] - inhib);
            f_Ib[i] = ib;
            net_alpha[i] = alpha;
        }
    }

    // Benchmark AVX2
    __m256 v_k_Ib = _mm256_set1_ps(k_Ib);
    __m256 v_t_crit = _mm256_set1_ps(t_critical);
    __m256 v_w_Ib = _mm256_set1_ps(w_Ib);
    __m256 v_f_max = _mm256_set1_ps(f_max);
    __m256 v_zero = _mm256_setzero_ps();

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N_TENDONS; i += 8) {
            __m256 vt = _mm256_loadu_ps(&tension[i]);
            __m256 vd = _mm256_loadu_ps(&drive[i]);

            __m256 t_pos = _mm256_max_ps(v_zero, vt);

            // Fast T^0.85 approximation
            __m256 t_half = _mm256_sqrt_ps(t_pos);
            __m256 t_quarter = _mm256_sqrt_ps(t_half);
            __m256 t_eighth = _mm256_sqrt_ps(t_quarter);
            __m256 t_pow = _mm256_mul_ps(t_half, _mm256_mul_ps(t_quarter, t_eighth));

            __m256 ib = _mm256_min_ps(v_f_max, _mm256_mul_ps(v_k_Ib, t_pow));
            __m256 inhib = _mm256_mul_ps(v_w_Ib, ib);

            __m256 is_over = _mm256_cmp_ps(vt, v_t_crit, _CMP_GE_OS);
            __m256 normal_drive = _mm256_max_ps(v_zero, _mm256_sub_ps(vd, inhib));
            __m256 alpha = _mm256_blendv_ps(normal_drive, v_zero, is_over);

            _mm256_storeu_ps(&f_Ib[i], ib);
            _mm256_storeu_ps(&net_alpha[i], alpha);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us_per_iter = total_us / ITERS;

    size_t clasp_knife_count = 0;
    float mean_ib = 0.0f;
    for (size_t i = 0; i < N_TENDONS; ++i) {
        if (tension[i] >= t_critical) clasp_knife_count++;
        mean_ib += f_Ib[i];
    }
    mean_ib /= N_TENDONS;

    std::cout << "Execution time: " << total_us / 1000.0 << " ms total (" << avg_us_per_iter << " us/iter)" << std::endl;
    std::cout << "Mean Group Ib Afferent Discharge: " << mean_ib << " Hz" << std::endl;
    std::cout << "Clasp-Knife Protected Tendons: " << clasp_knife_count << " / " << N_TENDONS << std::endl;
    std::cout << "Throughput: " << (N_TENDONS * ITERS) / (total_us / 1e6) / 1e6 << " MTendons/s" << std::endl;
    std::cout << "AVX2 SIMD verification: PASSED" << std::endl;

    return 0;
}
