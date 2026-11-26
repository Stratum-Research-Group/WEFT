/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: Pericyte Capillary Constriction & Ischemic No-Reflow
 * Target #412
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_CAPILLARIES = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_pericyte_no_reflow_avx2(
    const float* __restrict ca_in,
    const float* __restrict baseline_d,
    const float* __restrict shed_rescue,
    float* __restrict out_diameter,
    float* __restrict out_flux,
    size_t n
) {
    const __m256 v_ca_thresh = _mm256_set1_ps(400.0f);
    const __m256 v_constrict_factor = _mm256_set1_ps(0.0075f);
    const __m256 v_rescue_weight = _mm256_set1_ps(0.02f);
    const __m256 v_min_d = _mm256_set1_ps(2.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_ca = _mm256_loadu_ps(&ca_in[i]);
        __m256 v_d0 = _mm256_loadu_ps(&baseline_d[i]);
        __m256 v_rescue = _mm256_loadu_ps(&shed_rescue[i]);

        // Diameter contraction: delta = -Ca * constrict_factor + rescue * rescue_weight
        __m256 v_delta_ca = _mm256_mul_ps(v_ca, v_constrict_factor);
        __m256 v_delta_d = _mm256_fmsub_ps(v_rescue, v_rescue_weight, v_delta_ca);
        __m256 v_d = _mm256_add_ps(v_d0, v_delta_d);
        v_d = _mm256_max_ps(v_d, v_min_d);

        // Relative Poiseuille flux: (d / d0)^4
        __m256 v_ratio = _mm256_div_ps(v_d, v_d0);
        __m256 v_ratio2 = _mm256_mul_ps(v_ratio, v_ratio);
        __m256 v_flux = _mm256_mul_ps(v_ratio2, v_ratio2);

        _mm256_storeu_ps(&out_diameter[i], v_d);
        _mm256_storeu_ps(&out_flux[i], v_flux);
    }
}

int main() {
    std::cout << "[Target #412] Starting AVX2/FMA Pericyte Capillary Constriction & No-Reflow Benchmark..." << std::endl;

    std::vector<float> ca_in(N_CAPILLARIES);
    std::vector<float> baseline_d(N_CAPILLARIES);
    std::vector<float> shed_rescue(N_CAPILLARIES);
    std::vector<float> out_diameter(N_CAPILLARIES);
    std::vector<float> out_flux(N_CAPILLARIES);

    for (size_t i = 0; i < N_CAPILLARIES; ++i) {
        ca_in[i] = 120.0f + (i % 400); // 120 to 520 nM
        baseline_d[i] = 5.5f;           // 5.5 um baseline capillary lumen
        shed_rescue[i] = 80.0f;         // 80% secretome rescue efficacy
    }

    // Warmup
    benchmark_pericyte_no_reflow_avx2(ca_in.data(), baseline_d.data(), shed_rescue.data(), out_diameter.data(), out_flux.data(), N_CAPILLARIES);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_pericyte_no_reflow_avx2(ca_in.data(), baseline_d.data(), shed_rescue.data(), out_diameter.data(), out_flux.data(), N_CAPILLARIES);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_mcapillaries_sec = (double(N_CAPILLARIES) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_CAPILLARIES << " capillaries." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_mcapillaries_sec << " Million capillaries/sec" << std::endl;
    std::cout << "Sample Capillary 0 Lumen: " << out_diameter[0] << " um | Perfusion Efficiency: " << out_flux[0] * 100.0f << "%" << std::endl;

    return 0;
}
