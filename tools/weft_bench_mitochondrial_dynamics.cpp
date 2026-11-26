/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: Presynaptic Mitochondrial Bioenergetics & ATP Homeostasis
 * Target #415
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_MITOCHONDRIA = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_mitochondria_avx2(
    const float* __restrict ca_stress_in,
    const float* __restrict shed_rescue,
    float* __restrict out_psi,
    float* __restrict out_atp,
    size_t n
) {
    const __m256 v_stress_thresh = _mm256_set1_ps(50.0f);
    const __m256 v_psi_base_stress = _mm256_set1_ps(100.0f);
    const __m256 v_psi_slope_rescue = _mm256_set1_ps(0.55f);
    const __m256 v_psi_base_norm = _mm256_set1_ps(165.0f);
    const __m256 v_psi_slope_stress = _mm256_set1_ps(0.2f);

    const __m256 v_atp_base_stress = _mm256_set1_ps(1.5f);
    const __m256 v_atp_slope_rescue = _mm256_set1_ps(0.02f);
    const __m256 v_atp_base_norm = _mm256_set1_ps(3.8f);
    const __m256 v_atp_slope_stress = _mm256_set1_ps(0.01f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_stress = _mm256_loadu_ps(&ca_stress_in[i]);
        __m256 v_shed = _mm256_loadu_ps(&shed_rescue[i]);

        __m256 v_is_stress = _mm256_cmp_ps(v_stress, v_stress_thresh, _CMP_GT_OS);

        // Delta Psi_m calculation:
        // Stress: 100 + shed * 0.55
        // Normal: 165 - stress * 0.2
        __m256 v_psi_stress = _mm256_fmadd_ps(v_shed, v_psi_slope_rescue, v_psi_base_stress);
        __m256 v_psi_norm = _mm256_fnmadd_ps(v_stress, v_psi_slope_stress, v_psi_base_norm);
        __m256 v_psi = _mm256_blendv_ps(v_psi_norm, v_psi_stress, v_is_stress);

        // ATP calculation:
        // Stress: 1.5 + shed * 0.02
        // Normal: 3.8 - stress * 0.01
        __m256 v_atp_stress = _mm256_fmadd_ps(v_shed, v_atp_slope_rescue, v_atp_base_stress);
        __m256 v_atp_norm = _mm256_fnmadd_ps(v_stress, v_atp_slope_stress, v_atp_base_norm);
        __m256 v_atp = _mm256_blendv_ps(v_atp_norm, v_atp_stress, v_is_stress);

        _mm256_storeu_ps(&out_psi[i], v_psi);
        _mm256_storeu_ps(&out_atp[i], v_atp);
    }
}

int main() {
    std::cout << "[Target #415] Starting AVX2/FMA Mitochondrial Bioenergetics & ATP Benchmark..." << std::endl;

    std::vector<float> ca_stress_in(N_MITOCHONDRIA);
    std::vector<float> shed_rescue(N_MITOCHONDRIA);
    std::vector<float> out_psi(N_MITOCHONDRIA);
    std::vector<float> out_atp(N_MITOCHONDRIA);

    for (size_t i = 0; i < N_MITOCHONDRIA; ++i) {
        ca_stress_in[i] = 15.0f + (i % 80); // 15 to 95 Ca2+ stress train
        shed_rescue[i] = 90.0f;             // 90% PGC-1alpha / mitoprotection rescue
    }

    // Warmup
    benchmark_mitochondria_avx2(ca_stress_in.data(), shed_rescue.data(), out_psi.data(), out_atp.data(), N_MITOCHONDRIA);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_mitochondria_avx2(ca_stress_in.data(), shed_rescue.data(), out_psi.data(), out_atp.data(), N_MITOCHONDRIA);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_mmitos_sec = (double(N_MITOCHONDRIA) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_MITOCHONDRIA << " mitochondria." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_mmitos_sec << " Million mitochondria/sec" << std::endl;
    std::cout << "Sample Mito 0 Delta Psi_m: " << out_psi[0] << " mV | Synaptosomal ATP: " << out_atp[0] << " mM" << std::endl;

    return 0;
}
