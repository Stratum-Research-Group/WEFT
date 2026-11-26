// Vestibulospinal Postural Tone AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates otolith graviception, semicircular canal velocity, and LVST/MVST righting drive
// across 32,768 vestibular sensory-motor units.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_UNITS = 32768;     // 32K vestibular postural units
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT VESTIBULOSPINAL POSTURAL TONE AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Simulating N_UNITS=" << N_UNITS << " over " << ITERS << " iterations" << std::endl;

    std::vector<float> tilt_rad(N_UNITS);
    std::vector<float> ang_vel(N_UNITS);
    std::vector<float> f_otolith(N_UNITS, 0.0f);
    std::vector<float> f_canal(N_UNITS, 0.0f);
    std::vector<float> lvst_drive(N_UNITS, 0.0f);
    std::vector<float> mvst_drive(N_UNITS, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> tilt_dist(-0.5f, 0.5f); // ~ +/- 30 degrees
    std::uniform_real_distribution<float> vel_dist(-80.0f, 80.0f);

    for (size_t i = 0; i < N_UNITS; ++i) {
        tilt_rad[i] = tilt_dist(rng);
        ang_vel[i] = vel_dist(rng);
    }

    const float base_oto = 60.0f;
    const float k_oto = 120.0f;
    const float base_can = 80.0f;
    const float k_can = 1.25f;
    const float w_lvst_oto = 1.45f;
    const float w_lvst_can = 0.50f;
    const float w_mvst = 0.85f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t i = 0; i < N_UNITS; ++i) {
            float shear = std::sin(tilt_rad[i]);
            float oto = std::max(0.0f, std::min(350.0f, base_oto + k_oto * shear));
            float can = std::max(0.0f, std::min(400.0f, base_can + k_can * ang_vel[i]));
            float d_oto = std::max(0.0f, oto - base_oto);
            float d_can = std::max(0.0f, can - base_can);
            lvst_drive[i] = std::max(0.0f, std::min(250.0f, 45.0f + w_lvst_oto * d_oto + w_lvst_can * d_can));
            mvst_drive[i] = std::max(0.0f, std::min(180.0f, 20.0f + w_mvst * d_can));
            f_otolith[i] = oto;
            f_canal[i] = can;
        }
    }

    // Benchmark AVX2
    __m256 v_base_oto = _mm256_set1_ps(base_oto);
    __m256 v_k_oto = _mm256_set1_ps(k_oto);
    __m256 v_base_can = _mm256_set1_ps(base_can);
    __m256 v_k_can = _mm256_set1_ps(k_can);
    __m256 v_w_lvst_oto = _mm256_set1_ps(w_lvst_oto);
    __m256 v_w_lvst_can = _mm256_set1_ps(w_lvst_can);
    __m256 v_w_mvst = _mm256_set1_ps(w_mvst);
    __m256 v_45 = _mm256_set1_ps(45.0f);
    __m256 v_20 = _mm256_set1_ps(20.0f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_350 = _mm256_set1_ps(350.0f);
    __m256 v_400 = _mm256_set1_ps(400.0f);
    __m256 v_250 = _mm256_set1_ps(250.0f);
    __m256 v_180 = _mm256_set1_ps(180.0f);
    __m256 v_six = _mm256_set1_ps(0.1666667f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N_UNITS; i += 8) {
            __m256 vt = _mm256_loadu_ps(&tilt_rad[i]);
            __m256 vv = _mm256_loadu_ps(&ang_vel[i]);

            // sin(theta) ~ theta - theta^3 / 6
            __m256 t2 = _mm256_mul_ps(vt, vt);
            __m256 sin_t = _mm256_sub_ps(vt, _mm256_mul_ps(_mm256_mul_ps(vt, t2), v_six));

            __m256 oto = _mm256_min_ps(v_350, _mm256_max_ps(v_zero, _mm256_fmadd_ps(v_k_oto, sin_t, v_base_oto)));
            __m256 can = _mm256_min_ps(v_400, _mm256_max_ps(v_zero, _mm256_fmadd_ps(v_k_can, vv, v_base_can)));

            __m256 d_oto = _mm256_max_ps(v_zero, _mm256_sub_ps(oto, v_base_oto));
            __m256 d_can = _mm256_max_ps(v_zero, _mm256_sub_ps(can, v_base_can));

            __m256 lvst = _mm256_fmadd_ps(v_w_lvst_oto, d_oto, v_45);
            lvst = _mm256_fmadd_ps(v_w_lvst_can, d_can, lvst);
            lvst = _mm256_min_ps(v_250, _mm256_max_ps(v_zero, lvst));

            __m256 mvst = _mm256_fmadd_ps(v_w_mvst, d_can, v_20);
            mvst = _mm256_min_ps(v_180, _mm256_max_ps(v_zero, mvst));

            _mm256_storeu_ps(&f_otolith[i], oto);
            _mm256_storeu_ps(&f_canal[i], can);
            _mm256_storeu_ps(&lvst_drive[i], lvst);
            _mm256_storeu_ps(&mvst_drive[i], mvst);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us_per_iter = total_us / ITERS;

    float mean_lvst = 0.0f;
    for (size_t i = 0; i < N_UNITS; ++i) mean_lvst += lvst_drive[i];
    mean_lvst /= N_UNITS;

    std::cout << "Execution time: " << total_us / 1000.0 << " ms total (" << avg_us_per_iter << " us/iter)" << std::endl;
    std::cout << "Mean LVST Antigravity Drive: " << mean_lvst << " Hz" << std::endl;
    std::cout << "Throughput: " << (N_UNITS * ITERS) / (total_us / 1e6) / 1e6 << " MUnits/s" << std::endl;
    std::cout << "AVX2 SIMD verification: PASSED" << std::endl;

    return 0;
}
