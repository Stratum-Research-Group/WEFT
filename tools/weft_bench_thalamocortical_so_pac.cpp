/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
 * Native AVX2/FMA Benchmark: Thalamocortical Spindle-Slow Oscillation PAC
 * Target #418
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

constexpr size_t N_CHANNELS = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_so_pac_avx2(
    const float* __restrict desync_in,
    const float* __restrict acoustic_in,
    float* __restrict out_plv,
    float* __restrict out_amp,
    float* __restrict out_memory,
    size_t n
) {
    const __m256 v_desync_thresh = _mm256_set1_ps(50.0f);
    const __m256 v_plv_base_desync = _mm256_set1_ps(0.25f);
    const __m256 v_plv_slope_acoustic = _mm256_set1_ps(0.005f);
    const __m256 v_plv_base_norm = _mm256_set1_ps(0.82f);
    const __m256 v_plv_slope_desync = _mm256_set1_ps(0.0015f);

    const __m256 v_amp_base_desync = _mm256_set1_ps(14.0f);
    const __m256 v_amp_slope_acoustic = _mm256_set1_ps(0.26f);
    const __m256 v_amp_base_norm = _mm256_set1_ps(48.0f);
    const __m256 v_amp_slope_desync = _mm256_set1_ps(0.1f);

    const __m256 v_mem_base_desync = _mm256_set1_ps(22.0f);
    const __m256 v_mem_slope_acoustic = _mm256_set1_ps(0.55f);
    const __m256 v_mem_base_norm = _mm256_set1_ps(88.0f);
    const __m256 v_mem_slope_desync = _mm256_set1_ps(0.12f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_desync = _mm256_loadu_ps(&desync_in[i]);
        __m256 v_acoustic = _mm256_loadu_ps(&acoustic_in[i]);

        __m256 v_is_desync = _mm256_cmp_ps(v_desync, v_desync_thresh, _CMP_GT_OS);

        // PLV:
        // Desync: 0.25 + acoustic * 0.005
        // Norm: 0.82 - desync * 0.0015
        __m256 v_plv_d = _mm256_fmadd_ps(v_acoustic, v_plv_slope_acoustic, v_plv_base_desync);
        __m256 v_plv_n = _mm256_fnmadd_ps(v_desync, v_plv_slope_desync, v_plv_base_norm);
        __m256 v_plv = _mm256_blendv_ps(v_plv_n, v_plv_d, v_is_desync);

        // Amplitude:
        // Desync: 14 + acoustic * 0.26
        // Norm: 48 - desync * 0.1
        __m256 v_amp_d = _mm256_fmadd_ps(v_acoustic, v_amp_slope_acoustic, v_amp_base_desync);
        __m256 v_amp_n = _mm256_fnmadd_ps(v_desync, v_amp_slope_desync, v_amp_base_norm);
        __m256 v_amp = _mm256_blendv_ps(v_amp_n, v_amp_d, v_is_desync);

        // Memory:
        // Desync: 22 + acoustic * 0.55
        // Norm: 88 - desync * 0.12
        __m256 v_mem_d = _mm256_fmadd_ps(v_acoustic, v_mem_slope_acoustic, v_mem_base_desync);
        __m256 v_mem_n = _mm256_fnmadd_ps(v_desync, v_mem_slope_desync, v_mem_base_norm);
        __m256 v_mem = _mm256_blendv_ps(v_mem_n, v_mem_d, v_is_desync);

        _mm256_storeu_ps(&out_plv[i], v_plv);
        _mm256_storeu_ps(&out_amp[i], v_amp);
        _mm256_storeu_ps(&out_memory[i], v_mem);
    }
}

int main() {
    std::cout << "[Target #418] Starting AVX2/FMA Thalamocortical SO-PAC Benchmark..." << std::endl;

    std::vector<float> desync_in(N_CHANNELS);
    std::vector<float> acoustic_in(N_CHANNELS);
    std::vector<float> out_plv(N_CHANNELS);
    std::vector<float> out_amp(N_CHANNELS);
    std::vector<float> out_memory(N_CHANNELS);

    for (size_t i = 0; i < N_CHANNELS; ++i) {
        desync_in[i] = 10.0f + (i % 85); // 10 to 95 neurodegenerative desync
        acoustic_in[i] = 90.0f;          // 90% closed-loop acoustic stimulation drive
    }

    // Warmup
    benchmark_so_pac_avx2(desync_in.data(), acoustic_in.data(), out_plv.data(), out_amp.data(), out_memory.data(), N_CHANNELS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        benchmark_so_pac_avx2(desync_in.data(), acoustic_in.data(), out_plv.data(), out_amp.data(), out_memory.data(), N_CHANNELS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_us = (total_ms * 1000.0) / ITERATIONS;
    double throughput_mchannels_sec = (double(N_CHANNELS) * ITERATIONS) / (total_ms / 1000.0) / 1e6;

    std::cout << "Done. Completed " << ITERATIONS << " iterations of " << N_CHANNELS << " electrophysiological channels." << std::endl;
    std::cout << "Execution time per iteration: " << std::fixed << std::setprecision(3) << per_iter_us << " us" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput_mchannels_sec << " Million channels/sec" << std::endl;
    std::cout << "Sample Channel 0 PLV: " << out_plv[0] << " | Spindle Amp: " << out_amp[0] << " uV | Memory Consolidation: " << out_memory[0] << "%" << std::endl;

    return 0;
}
