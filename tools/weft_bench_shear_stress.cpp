/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: Endothelial Shear-Stress Mechanotransduction & eNOS Phosphorylation
 * Target #413
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_ENDOTHELIAL_CELLS = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_shear_stress_avx2(
    const float* __restrict shear_in,
    const float* __restrict shed_rescue,
    float* __restrict out_phos,
    float* __restrict out_no,
    size_t n
) {
    const __m256 v_shear_thresh = _mm256_set1_ps(12.0f);
    const __m256 v_laminar_phos_base = _mm256_set1_ps(70.0f);
    const __m256 v_laminar_slope = _mm256_set1_ps(0.8f);
    const __m256 v_disturbed_phos_base = _mm256_set1_ps(25.0f);
    const __m256 v_rescue_phos_slope = _mm256_set1_ps(0.45f);

    const __m256 v_laminar_no_base = _mm256_set1_ps(1.8f);
    const __m256 v_laminar_no_slope = _mm256_set1_ps(0.04f);
    const __m256 v_disturbed_no_base = _mm256_set1_ps(0.6f);
    const __m256 v_rescue_no_slope = _mm256_set1_ps(0.015f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_tau = _mm256_loadu_ps(&shear_in[i]);
        __m256 v_shed = _mm256_loadu_ps(&shed_rescue[i]);

        __m256 v_is_laminar = _mm256_cmp_ps(v_tau, v_shear_thresh, _CMP_GE_OS);

        // Phosphorylation computation:
        // Laminar: 70 + tau * 0.8
        // Disturbed: 25 + shed * 0.45
        __m256 v_phos_lam = _mm256_fmadd_ps(v_tau, v_laminar_slope, v_laminar_phos_base);
        __m256 v_phos_dist = _mm256_fmadd_ps(v_shed, v_rescue_phos_slope, v_disturbed_phos_base);
        __m256 v_phos = _mm256_blendv_ps(v_phos_dist, v_phos_lam, v_is_laminar);

        // Nitric Oxide computation:
        // Laminar: 1.8 + tau * 0.04
        // Disturbed: 0.6 + shed * 0.015
        __m256 v_no_lam = _mm256_fmadd_ps(v_tau, v_laminar_no_slope, v_laminar_no_base);
        __m256 v_no_dist = _mm256_fmadd_ps(v_shed, v_rescue_no_slope, v_disturbed_no_base);
        __m256 v_no = _mm256_blendv_ps(v_no_dist, v_no_lam, v_is_laminar);

        _mm256_storeu_ps(&out_phos[i], v_phos);
        _mm256_storeu_ps(&out_no[i], v_no);
    }
}

int main() {
    std::cout << "[Target #413] Starting AVX2/FMA Endothelial Shear-Stress Mechanotransduction Benchmark..." << std::endl;

    std::vector<float> shear_in(N_ENDOTHELIAL_CELLS);
    std::vector<float> shed_rescue(N_ENDOTHELIAL_CELLS);
    std::vector<float> out_phos(N_ENDOTHELIAL_CELLS);
    std::vector<float> out_no(N_ENDOTHELIAL_CELLS);

    for (size_t i = 0; i < N_ENDOTHELIAL_CELLS; ++i) {
        shear_in[i] = 2.0f + (i % 25);  // 2 to 27 dyn/cm2
        shed_rescue[i] = 85.0f;         // 85% secretome rescue efficacy
    }

    // Warmup
    benchmark_shear_stress_avx2(shear_in.data(), shed_rescue.data(), out_phos.data(), out_no.data(), N_ENDOTHELIAL_CELLS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_shear_stress_avx2(shear_in.data(), shed_rescue.data(), out_phos.data(), out_no.data(), N_ENDOTHELIAL_CELLS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_mcells_sec = (double(N_ENDOTHELIAL_CELLS) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_ENDOTHELIAL_CELLS << " endothelial cells." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_mcells_sec << " Million cells/sec" << std::endl;
    std::cout << "Sample Cell 0 eNOS Phos: " << out_phos[0] << "% | Bioavailable NO: " << out_no[0] << " uM" << std::endl;

    return 0;
}
