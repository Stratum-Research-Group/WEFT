// tools/weft_bench_purkinje_complex_spike.cpp
// Native AVX2/FMA SIMD benchmark for Cerebellar Purkinje Complex Spike & Climbing Fiber LTD Kinetics
// Computes dendritic Ca2+ transients, mGluR1/IP3/DAG activation, parallel fiber AMPA receptor internalization, and prediction error

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr size_t N_CHANNELS = 65536;
static constexpr int N_ITERATIONS = 1000;

void benchmark_purkinje_complex_spike_avx2(
    const float* __restrict__ cf_freq_hz,
    const float* __restrict__ pf_drive_pct,
    float* __restrict__ out_ca,
    float* __restrict__ out_mglur1,
    float* __restrict__ out_ltd,
    float* __restrict__ out_error,
    size_t n
) {
    const __m256 v_cf_ca_slope = _mm256_set1_ps(2.2f / 4.0f);
    const __m256 v_pf_ca_slope = _mm256_set1_ps(0.10f / 100.0f);
    const __m256 v_pf_mglur_slope = _mm256_set1_ps(0.042f);
    const __m256 v_ltd_scale = _mm256_set1_ps(38.0f / (2.0f * 3.0f));
    const __m256 v_min_ltd = _mm256_set1_ps(5.0f);
    const __m256 v_max_ltd = _mm256_set1_ps(65.0f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_err_slope = _mm256_set1_ps(0.85f / 50.0f);
    const __m256 v_min_err = _mm256_set1_ps(0.05f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_cf = _mm256_loadu_ps(cf_freq_hz + i);
        __m256 v_pf = _mm256_loadu_ps(pf_drive_pct + i);

        // ca = cf * 0.55 + pf * 0.001
        __m256 v_ca_part = _mm256_mul_ps(v_cf, v_cf_ca_slope);
        __m256 v_ca = _mm256_fmadd_ps(v_pf, v_pf_ca_slope, v_ca_part);
        _mm256_storeu_ps(out_ca + i, v_ca);

        // mglur1 = pf * 0.042
        __m256 v_mglur = _mm256_mul_ps(v_pf, v_pf_mglur_slope);
        _mm256_storeu_ps(out_mglur1 + i, v_mglur);

        // ltd = clamp(ca * mglur1 * 6.333, 5.0, 65.0)
        __m256 v_raw_ltd = _mm256_mul_ps(_mm256_mul_ps(v_ca, v_mglur), v_ltd_scale);
        __m256 v_ltd = _mm256_max_ps(v_min_ltd, _mm256_min_ps(v_max_ltd, v_raw_ltd));
        _mm256_storeu_ps(out_ltd + i, v_ltd);

        // error = max(0.05, 1.0 - ltd * 0.017)
        __m256 v_err_drop = _mm256_mul_ps(v_ltd, v_err_slope);
        __m256 v_err_raw = _mm256_sub_ps(v_one, v_err_drop);
        __m256 v_err = _mm256_max_ps(v_min_err, v_err_raw);
        _mm256_storeu_ps(out_error + i, v_err);
    }
}

int main() {
    std::vector<float> cf(N_CHANNELS, 2.5f);
    std::vector<float> pf(N_CHANNELS, 80.0f);
    std::vector<float> ca(N_CHANNELS);
    std::vector<float> mglur(N_CHANNELS);
    std::vector<float> ltd(N_CHANNELS);
    std::vector<float> error(N_CHANNELS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        benchmark_purkinje_complex_spike_avx2(
            cf.data(), pf.data(),
            ca.data(), mglur.data(), ltd.data(), error.data(),
            N_CHANNELS
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERATIONS; ++i) {
        benchmark_purkinje_complex_spike_avx2(
            cf.data(), pf.data(),
            ca.data(), mglur.data(), ltd.data(), error.data(),
            N_CHANNELS
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / N_ITERATIONS;
    double throughput_mch_s = (static_cast<double>(N_CHANNELS) / (avg_us * 1e-6)) / 1e6;

    std::cout << "[AVX2-FMA Benchmark] Cerebellar Purkinje Complex Spike & LTD Kinetics" << std::endl;
    std::cout << "Channels: " << N_CHANNELS << " | Iterations: " << N_ITERATIONS << std::endl;
    std::cout << "Avg latency per iteration: " << avg_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput_mch_s << " Million channels/sec" << std::endl;
    std::cout << "Sample verification [Channel 0]:" << std::endl;
    std::cout << "  Climbing Fiber: " << cf[0] << " Hz | Parallel Fiber: " << pf[0] << "%" << std::endl;
    std::cout << "  Dendritic Ca2+: " << ca[0] << " uM | mGluR1 Ratio: " << mglur[0] << "x" << std::endl;
    std::cout << "  Parallel Fiber LTD: " << ltd[0] << "% | Motor Prediction Error: " << error[0] << std::endl;

    return 0;
}
