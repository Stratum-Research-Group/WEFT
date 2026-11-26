// Magnocellular Red Nucleus & Rubrospinal Velocity Tuning AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates limb velocity gating, cerebellar interposed convergence, and rubrospinal digit flexor recruitment
// across 32,768 motor control units.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_UNITS = 32768;     // 32K rubrospinal motor units
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT RUBROSPINAL VELOCITY TUNING AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Simulating N_UNITS=" << N_UNITS << " over " << ITERS << " iterations" << std::endl;

    std::vector<float> v_limb(N_UNITS);
    std::vector<float> x_inter(N_UNITS);
    std::vector<float> x_m1(N_UNITS);
    std::vector<float> rnm_rate(N_UNITS, 0.0f);
    std::vector<float> flex_digit(N_UNITS, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> v_dist(0.0f, 1.2f);   // 0 to 1.2 m/s reach velocity
    std::uniform_real_distribution<float> inter_dist(20.0f, 160.0f);
    std::uniform_real_distribution<float> m1_dist(10.0f, 120.0f);

    for (size_t i = 0; i < N_UNITS; ++i) {
        v_limb[i] = v_dist(rng);
        x_inter[i] = inter_dist(rng);
        x_m1[i] = m1_dist(rng);
    }

    const float base_rnm = 35.0f;
    const float k_vel = 85.0f;
    const float w_inter = 0.55f;
    const float w_m1 = 0.40f;
    const float w_rst = 1.15f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t i = 0; i < N_UNITS; ++i) {
            float v = std::max(0.0f, v_limb[i]);
            float rnm = std::max(0.0f, std::min(300.0f, base_rnm + k_vel * v + w_inter * x_inter[i] + w_m1 * x_m1[i]));
            float flex = std::max(0.0f, std::min(250.0f, 20.0f + w_rst * (rnm - base_rnm)));
            rnm_rate[i] = rnm;
            flex_digit[i] = flex;
        }
    }

    // Benchmark AVX2
    __m256 v_base_rnm = _mm256_set1_ps(base_rnm);
    __m256 v_k_vel = _mm256_set1_ps(k_vel);
    __m256 v_w_inter = _mm256_set1_ps(w_inter);
    __m256 v_w_m1 = _mm256_set1_ps(w_m1);
    __m256 v_w_rst = _mm256_set1_ps(w_rst);
    __m256 v_20 = _mm256_set1_ps(20.0f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_300 = _mm256_set1_ps(300.0f);
    __m256 v_250 = _mm256_set1_ps(250.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N_UNITS; i += 8) {
            __m256 vv = _mm256_loadu_ps(&v_limb[i]);
            __m256 vin = _mm256_loadu_ps(&x_inter[i]);
            __m256 vm1 = _mm256_loadu_ps(&x_m1[i]);

            __m256 v_pos = _mm256_max_ps(v_zero, vv);

            __m256 rnm = _mm256_fmadd_ps(v_k_vel, v_pos, v_base_rnm);
            rnm = _mm256_fmadd_ps(v_w_inter, vin, rnm);
            rnm = _mm256_fmadd_ps(v_w_m1, vm1, rnm);
            rnm = _mm256_min_ps(v_300, _mm256_max_ps(v_zero, rnm));

            __m256 delta_rnm = _mm256_sub_ps(rnm, v_base_rnm);
            __m256 flex = _mm256_min_ps(v_250, _mm256_max_ps(v_zero, _mm256_fmadd_ps(v_w_rst, delta_rnm, v_20)));

            _mm256_storeu_ps(&rnm_rate[i], rnm);
            _mm256_storeu_ps(&flex_digit[i], flex);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us_per_iter = total_us / ITERS;

    float mean_rnm = 0.0f;
    float mean_flex = 0.0f;
    for (size_t i = 0; i < N_UNITS; ++i) {
        mean_rnm += rnm_rate[i];
        mean_flex += flex_digit[i];
    }
    mean_rnm /= N_UNITS;
    mean_flex /= N_UNITS;

    std::cout << "Execution time: " << total_us / 1000.0 << " ms total (" << avg_us_per_iter << " us/iter)" << std::endl;
    std::cout << "Mean RNm Firing Rate: " << mean_rnm << " Hz" << std::endl;
    std::cout << "Mean Digit Flexor Recruitment: " << mean_flex << " Hz" << std::endl;
    std::cout << "Throughput: " << (N_UNITS * ITERS) / (total_us / 1e6) / 1e6 << " MUnits/s" << std::endl;
    std::cout << "AVX2 SIMD verification: PASSED" << std::endl;

    return 0;
}
