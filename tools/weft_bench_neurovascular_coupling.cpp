/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: Neurovascular Coupling & BK_Ca Vasomotor Simulation
 * Target #411
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_MICROVESSELS = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_neurovascular_avx2(
    const float* __restrict ca_in,
    const float* __restrict baseline_r,
    const float* __restrict eet_eff,
    float* __restrict out_radius,
    float* __restrict out_flux,
    size_t n
) {
    const __m256 v_ca_thresh = _mm256_set1_ps(500.0f);
    const __m256 v_k_scale = _mm256_set1_ps(0.025f);
    const __m256 v_eet_weight = _mm256_set1_ps(0.05f);
    const __m256 v_poiseuille_exp = _mm256_set1_ps(4.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_ca = _mm256_loadu_ps(&ca_in[i]);
        __m256 v_r0 = _mm256_loadu_ps(&baseline_r[i]);
        __m256 v_eet = _mm256_loadu_ps(&eet_eff[i]);

        // Arteriolar dilation delta: dr = (EET * 0.05) + (Ca * 0.025 if Ca < 500 else -Ca*0.01)
        __m256 v_mask = _mm256_cmp_ps(v_ca, v_ca_thresh, _CMP_LT_OS);
        __m256 v_dil = _mm256_mul_ps(v_ca, v_k_scale);
        __m256 v_constrict = _mm256_mul_ps(v_ca, _mm256_set1_ps(-0.015f));
        __m256 v_delta_ca = _mm256_blendv_ps(v_constrict, v_dil, v_mask);

        __m256 v_delta_r = _mm256_fmadd_ps(v_eet, v_eet_weight, v_delta_ca);
        __m256 v_r = _mm256_add_ps(v_r0, v_delta_r);

        // Poiseuille approximation: flux ~ (r / r0)^4
        __m256 v_ratio = _mm256_div_ps(v_r, v_r0);
        __m256 v_ratio2 = _mm256_mul_ps(v_ratio, v_ratio);
        __m256 v_flux = _mm256_mul_ps(v_ratio2, v_ratio2);

        _mm256_storeu_ps(&out_radius[i], v_r);
        _mm256_storeu_ps(&out_flux[i], v_flux);
    }
}

int main() {
    std::cout << "[Target #411] Starting AVX2/FMA Neurovascular Coupling & BK_Ca Benchmark..." << std::endl;

    std::vector<float> ca_in(N_MICROVESSELS);
    std::vector<float> baseline_r(N_MICROVESSELS);
    std::vector<float> eet_eff(N_MICROVESSELS);
    std::vector<float> out_radius(N_MICROVESSELS);
    std::vector<float> out_flux(N_MICROVESSELS);

    for (size_t i = 0; i < N_MICROVESSELS; ++i) {
        ca_in[i] = 250.0f + (i % 250); // 250 to 500 nM
        baseline_r[i] = 20.0f + (i % 10) * 0.5f; // 20 to 25 um
        eet_eff[i] = 50.0f + (i % 50);
    }

    // Warmup
    benchmark_neurovascular_avx2(ca_in.data(), baseline_r.data(), eet_eff.data(), out_radius.data(), out_flux.data(), N_MICROVESSELS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_neurovascular_avx2(ca_in.data(), baseline_r.data(), eet_eff.data(), out_radius.data(), out_flux.data(), N_MICROVESSELS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_mvessels_sec = (double(N_MICROVESSELS) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_MICROVESSELS << " microvessels." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_mvessels_sec << " Million vessels/sec" << std::endl;
    std::cout << "Sample Vessel 0 Radius: " << out_radius[0] << " um | Perfusion Flux: " << out_flux[0] * 100.0f << "%" << std::endl;

    return 0;
}
