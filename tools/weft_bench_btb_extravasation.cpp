/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: Blood-Tumor Barrier (BTB) Heterogeneity & Convective Extravasation
 * Target #416
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_VESSELS = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_btb_extravasation_avx2(
    const float* __restrict tumor_stage_in,
    const float* __restrict rescue_in,
    float* __restrict out_ifp,
    float* __restrict out_flux,
    float* __restrict out_delivery,
    size_t n
) {
    const __m256 v_stage_thresh = _mm256_set1_ps(50.0f);
    const __m256 v_ifp_base_tumor = _mm256_set1_ps(28.0f);
    const __m256 v_ifp_slope_rescue = _mm256_set1_ps(0.22f);
    const __m256 v_ifp_base_norm = _mm256_set1_ps(4.0f);
    const __m256 v_ifp_slope_stage = _mm256_set1_ps(0.1f);

    const __m256 v_flux_base_tumor = _mm256_set1_ps(1.5f);
    const __m256 v_flux_slope_rescue = _mm256_set1_ps(0.055f);
    const __m256 v_flux_base_norm = _mm256_set1_ps(7.0f);
    const __m256 v_flux_slope_stage = _mm256_set1_ps(0.02f);

    const __m256 v_del_base_tumor = _mm256_set1_ps(12.0f);
    const __m256 v_del_slope_rescue = _mm256_set1_ps(0.52f);
    const __m256 v_del_base_norm = _mm256_set1_ps(65.0f);
    const __m256 v_del_slope_stage = _mm256_set1_ps(0.15f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_stage = _mm256_loadu_ps(&tumor_stage_in[i]);
        __m256 v_rescue = _mm256_loadu_ps(&rescue_in[i]);

        __m256 v_is_tumor = _mm256_cmp_ps(v_stage, v_stage_thresh, _CMP_GT_OS);

        // IFP:
        // Tumor: 28 - rescue * 0.22
        // Norm: 4 + stage * 0.1
        __m256 v_ifp_tum = _mm256_fnmadd_ps(v_rescue, v_ifp_slope_rescue, v_ifp_base_tumor);
        __m256 v_ifp_nor = _mm256_fmadd_ps(v_stage, v_ifp_slope_stage, v_ifp_base_norm);
        __m256 v_ifp = _mm256_blendv_ps(v_ifp_nor, v_ifp_tum, v_is_tumor);

        // Flux:
        // Tumor: 1.5 + rescue * 0.055
        // Norm: 7.0 - stage * 0.02
        __m256 v_flux_tum = _mm256_fmadd_ps(v_rescue, v_flux_slope_rescue, v_flux_base_tumor);
        __m256 v_flux_nor = _mm256_fnmadd_ps(v_stage, v_flux_slope_stage, v_flux_base_norm);
        __m256 v_flux = _mm256_blendv_ps(v_flux_nor, v_flux_tum, v_is_tumor);

        // Delivery:
        // Tumor: 12 + rescue * 0.52
        // Norm: 65 - stage * 0.15
        __m256 v_del_tum = _mm256_fmadd_ps(v_rescue, v_del_slope_rescue, v_del_base_tumor);
        __m256 v_del_nor = _mm256_fnmadd_ps(v_stage, v_del_slope_stage, v_del_base_norm);
        __m256 v_del = _mm256_blendv_ps(v_del_nor, v_del_tum, v_is_tumor);

        _mm256_storeu_ps(&out_ifp[i], v_ifp);
        _mm256_storeu_ps(&out_flux[i], v_flux);
        _mm256_storeu_ps(&out_delivery[i], v_del);
    }
}

int main() {
    std::cout << "[Target #416] Starting AVX2/FMA BTB Extravasation & IFP Benchmark..." << std::endl;

    std::vector<float> tumor_stage_in(N_VESSELS);
    std::vector<float> rescue_in(N_VESSELS);
    std::vector<float> out_ifp(N_VESSELS);
    std::vector<float> out_flux(N_VESSELS);
    std::vector<float> out_delivery(N_VESSELS);

    for (size_t i = 0; i < N_VESSELS; ++i) {
        tumor_stage_in[i] = 10.0f + (i % 85); // 10 to 95 tumor vascular progression
        rescue_in[i] = 90.0f;                 // 90% Tie2 normalization + FUS cavitation
    }

    // Warmup
    benchmark_btb_extravasation_avx2(tumor_stage_in.data(), rescue_in.data(), out_ifp.data(), out_flux.data(), out_delivery.data(), N_VESSELS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_btb_extravasation_avx2(tumor_stage_in.data(), rescue_in.data(), out_ifp.data(), out_flux.data(), out_delivery.data(), N_VESSELS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_mvessels_sec = (double(N_VESSELS) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_VESSELS << " microvessels." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_mvessels_sec << " Million microvessels/sec" << std::endl;
    std::cout << "Sample Vessel 0 IFP: " << out_ifp[0] << " mmHg | Flux: " << out_flux[0] << " uL/min | Nanocarrier Delivery: " << out_delivery[0] << "%" << std::endl;

    return 0;
}
