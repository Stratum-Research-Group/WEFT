/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: Dural Meningeal Lymphatic Drainage & Deep Cervical Clearance
 * Target #417
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_LYMPHATICS = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_meningeal_lymphatics_avx2(
    const float* __restrict aging_in,
    const float* __restrict vegfc_in,
    float* __restrict out_velocity,
    float* __restrict out_efflux,
    float* __restrict out_clearance,
    size_t n
) {
    const __m256 v_aging_thresh = _mm256_set1_ps(50.0f);
    const __m256 v_vel_base_aged = _mm256_set1_ps(1.5f);
    const __m256 v_vel_slope_vegfc = _mm256_set1_ps(0.035f);
    const __m256 v_vel_base_norm = _mm256_set1_ps(5.5f);
    const __m256 v_vel_slope_aging = _mm256_set1_ps(0.015f);

    const __m256 v_eff_base_aged = _mm256_set1_ps(0.8f);
    const __m256 v_eff_slope_vegfc = _mm256_set1_ps(0.022f);
    const __m256 v_eff_base_norm = _mm256_set1_ps(3.2f);
    const __m256 v_eff_slope_aging = _mm256_set1_ps(0.008f);

    const __m256 v_clr_base_aged = _mm256_set1_ps(20.0f);
    const __m256 v_clr_slope_vegfc = _mm256_set1_ps(0.55f);
    const __m256 v_clr_base_norm = _mm256_set1_ps(80.0f);
    const __m256 v_clr_slope_aging = _mm256_set1_ps(0.2f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_age = _mm256_loadu_ps(&aging_in[i]);
        __m256 v_vegfc = _mm256_loadu_ps(&vegfc_in[i]);

        __m256 v_is_aged = _mm256_cmp_ps(v_age, v_aging_thresh, _CMP_GT_OS);

        // Velocity:
        // Aged: 1.5 + vegfc * 0.035
        // Norm: 5.5 - age * 0.015
        __m256 v_v_aged = _mm256_fmadd_ps(v_vegfc, v_vel_slope_vegfc, v_vel_base_aged);
        __m256 v_v_norm = _mm256_fnmadd_ps(v_age, v_vel_slope_aging, v_vel_base_norm);
        __m256 v_v = _mm256_blendv_ps(v_v_norm, v_v_aged, v_is_aged);

        // Efflux:
        // Aged: 0.8 + vegfc * 0.022
        // Norm: 3.2 - age * 0.008
        __m256 v_e_aged = _mm256_fmadd_ps(v_vegfc, v_eff_slope_vegfc, v_eff_base_aged);
        __m256 v_e_norm = _mm256_fnmadd_ps(v_age, v_eff_slope_aging, v_eff_base_norm);
        __m256 v_e = _mm256_blendv_ps(v_e_norm, v_e_aged, v_is_aged);

        // Clearance:
        // Aged: 20 + vegfc * 0.55
        // Norm: 80 - age * 0.2
        __m256 v_c_aged = _mm256_fmadd_ps(v_vegfc, v_clr_slope_vegfc, v_clr_base_aged);
        __m256 v_c_norm = _mm256_fnmadd_ps(v_age, v_clr_slope_aging, v_clr_base_norm);
        __m256 v_c = _mm256_blendv_ps(v_c_norm, v_c_aged, v_is_aged);

        _mm256_storeu_ps(&out_velocity[i], v_v);
        _mm256_storeu_ps(&out_efflux[i], v_e);
        _mm256_storeu_ps(&out_clearance[i], v_c);
    }
}

int main() {
    std::cout << "[Target #417] Starting AVX2/FMA Meningeal Lymphatics & dCLN Clearance Benchmark..." << std::endl;

    std::vector<float> aging_in(N_LYMPHATICS);
    std::vector<float> vegfc_in(N_LYMPHATICS);
    std::vector<float> out_velocity(N_LYMPHATICS);
    std::vector<float> out_efflux(N_LYMPHATICS);
    std::vector<float> out_clearance(N_LYMPHATICS);

    for (size_t i = 0; i < N_LYMPHATICS; ++i) {
        aging_in[i] = 10.0f + (i % 85); // 10 to 95 aging/fibrosis
        vegfc_in[i] = 90.0f;            // 90% VEGF-C lymphangiogenic drive
    }

    // Warmup
    benchmark_meningeal_lymphatics_avx2(aging_in.data(), vegfc_in.data(), out_velocity.data(), out_efflux.data(), out_clearance.data(), N_LYMPHATICS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_meningeal_lymphatics_avx2(aging_in.data(), vegfc_in.data(), out_velocity.data(), out_efflux.data(), out_clearance.data(), N_LYMPHATICS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_mlymph_sec = (double(N_LYMPHATICS) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_LYMPHATICS << " lymphatic vessels." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_mlymph_sec << " Million lymphatic vessels/sec" << std::endl;
    std::cout << "Sample Vessel 0 Velocity: " << out_velocity[0] << " um/s | dCLN Efflux: " << out_efflux[0] << " uL/min | Clearance: " << out_clearance[0] << "%" << std::endl;

    return 0;
}
