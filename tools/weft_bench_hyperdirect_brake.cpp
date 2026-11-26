#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: Cortico-Subthalamic Hyperdirect Motor Brake
// Benchmarks 32,768 subthalamo-pallidal motor gating units.

void benchmark_hyperdirect_brake_avx2(size_t n, size_t iterations) {
    std::vector<float> v_stop(n);
    std::vector<float> v_ongoing(n);
    std::vector<float> stn_rate_out(n, 0.0f);
    std::vector<float> gpi_rate_out(n, 0.0f);
    std::vector<float> brake_out(n, 0.0f);
    std::vector<float> aborted_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        v_stop[i] = static_cast<float>(i) / static_cast<float>(n); // 0.0 to 1.0 stop signal
        v_ongoing[i] = 1.0f;
    }

    const __m256 v_fstn_base = _mm256_set1_ps(20.0f);
    const __m256 v_kstn = _mm256_set1_ps(180.0f);
    const __m256 v_max_stn = _mm256_set1_ps(400.0f);
    const __m256 v_fgpi_base = _mm256_set1_ps(60.0f);
    const __m256 v_kgpi = _mm256_set1_ps(0.75f);
    const __m256 v_max_gpi = _mm256_set1_ps(350.0f);
    const __m256 v_inv150 = _mm256_set1_ps(1.0f / 150.0f);
    const __m256 v_gpi_abort_th = _mm256_set1_ps(120.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 vs = _mm256_loadu_ps(&v_stop[i]);

            // STN rate: min(400.0, 20.0 + 180.0 * v_stop)
            __m256 stn = _mm256_add_ps(v_fstn_base, _mm256_mul_ps(v_kstn, vs));
            stn = _mm256_min_ps(stn, v_max_stn);

            // GPi rate: min(350.0, 60.0 + 0.75 * (stn - 20.0))
            __m256 diff_stn = _mm256_sub_ps(stn, v_fstn_base);
            __m256 gpi = _mm256_add_ps(v_fgpi_base, _mm256_mul_ps(v_kgpi, diff_stn));
            gpi = _mm256_min_ps(gpi, v_max_gpi);

            // Thalamic brake fraction: clamp((gpi - 60.0) / 150.0, 0.0, 1.0)
            __m256 diff_gpi = _mm256_sub_ps(gpi, v_fgpi_base);
            __m256 raw_brake = _mm256_mul_ps(diff_gpi, v_inv150);
            __m256 brake = _mm256_max_ps(v_zero, _mm256_min_ps(v_one, raw_brake));

            // Abort state: gpi >= 120.0
            __m256 is_abort_mask = _mm256_cmp_ps(gpi, v_gpi_abort_th, _CMP_GE_OQ);
            __m256 is_abort = _mm256_blendv_ps(v_zero, v_one, is_abort_mask);

            _mm256_storeu_ps(&stn_rate_out[i], stn);
            _mm256_storeu_ps(&gpi_rate_out[i], gpi);
            _mm256_storeu_ps(&brake_out[i], brake);
            _mm256_storeu_ps(&aborted_out[i], is_abort);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] Hyperdirect Gating Units: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MGatingUnits/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: Stop=" << v_stop[n/2] << " => STN=" 
              << stn_rate_out[n/2] << " Hz, GPi=" << gpi_rate_out[n/2] << " Hz, Brake=" 
              << brake_out[n/2] * 100.0f << "%, Aborted=" << (aborted_out[n/2] > 0.5f ? "YES" : "NO") << std::endl;
}

int main() {
    benchmark_hyperdirect_brake_avx2(32768, 5000);
    return 0;
}
