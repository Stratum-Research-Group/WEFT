/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: OPC Differentiation & Saltatory Conduction Velocity
 * Target #414
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_AXONS = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_opc_conduction_avx2(
    const float* __restrict insult_in,
    const float* __restrict shed_rescue,
    float* __restrict out_thickness,
    float* __restrict out_velocity,
    size_t n
) {
    const __m256 v_insult_thresh = _mm256_set1_ps(50.0f);
    const __m256 v_mbp_base_demyel = _mm256_set1_ps(25.0f);
    const __m256 v_mbp_slope_rescue = _mm256_set1_ps(1.25f);
    const __m256 v_mbp_base_norm = _mm256_set1_ps(150.0f);
    const __m256 v_mbp_slope_insult = _mm256_set1_ps(0.3f);

    const __m256 v_vel_base_demyel = _mm256_set1_ps(15.0f);
    const __m256 v_vel_slope_rescue = _mm256_set1_ps(0.6f);
    const __m256 v_vel_base_norm = _mm256_set1_ps(80.0f);
    const __m256 v_vel_slope_insult = _mm256_set1_ps(0.2f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_insult = _mm256_loadu_ps(&insult_in[i]);
        __m256 v_shed = _mm256_loadu_ps(&shed_rescue[i]);

        __m256 v_is_demyel = _mm256_cmp_ps(v_insult, v_insult_thresh, _CMP_GT_OS);

        // MBP thickness:
        // Demyel: 25 + shed * 1.25
        // Norm: 150 - insult * 0.3
        __m256 v_mbp_demyel = _mm256_fmadd_ps(v_shed, v_mbp_slope_rescue, v_mbp_base_demyel);
        __m256 v_mbp_norm = _mm256_fnmadd_ps(v_insult, v_mbp_slope_insult, v_mbp_base_norm);
        __m256 v_mbp = _mm256_blendv_ps(v_mbp_norm, v_mbp_demyel, v_is_demyel);

        // Velocity:
        // Demyel: 15 + shed * 0.6
        // Norm: 80 - insult * 0.2
        __m256 v_vel_demyel = _mm256_fmadd_ps(v_shed, v_vel_slope_rescue, v_vel_base_demyel);
        __m256 v_vel_norm = _mm256_fnmadd_ps(v_insult, v_vel_slope_insult, v_vel_base_norm);
        __m256 v_vel = _mm256_blendv_ps(v_vel_norm, v_vel_demyel, v_is_demyel);

        _mm256_storeu_ps(&out_thickness[i], v_mbp);
        _mm256_storeu_ps(&out_velocity[i], v_vel);
    }
}

int main() {
    std::cout << "[Target #414] Starting AVX2/FMA OPC Differentiation & Conduction Benchmark..." << std::endl;

    std::vector<float> insult_in(N_AXONS);
    std::vector<float> shed_rescue(N_AXONS);
    std::vector<float> out_thickness(N_AXONS);
    std::vector<float> out_velocity(N_AXONS);

    for (size_t i = 0; i < N_AXONS; ++i) {
        insult_in[i] = 10.0f + (i % 85); // 10 to 95 cuprizone insult
        shed_rescue[i] = 90.0f;          // 90% BDNF/IGF-1 secretome rescue
    }

    // Warmup
    benchmark_opc_conduction_avx2(insult_in.data(), shed_rescue.data(), out_thickness.data(), out_velocity.data(), N_AXONS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_opc_conduction_avx2(insult_in.data(), shed_rescue.data(), out_thickness.data(), out_velocity.data(), N_AXONS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_maxons_sec = (double(N_AXONS) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_AXONS << " axons." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_maxons_sec << " Million axons/sec" << std::endl;
    std::cout << "Sample Axon 0 Sheath: " << out_thickness[0] << " nm | Saltatory Velocity: " << out_velocity[0] << " m/s" << std::endl;

    return 0;
}
