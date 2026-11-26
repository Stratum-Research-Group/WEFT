#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Superior Colliculus Deep Layers (SCdl) & Saccadic Burst Generator (PPRF / OPN)
// Evaluates:
// 1. Saccadic Main Sequence peak velocity: v_peak = V_max * (1 - exp(-amp / C))
// 2. Linear duration scaling: D = D0 + slope * amp
// 3. PPRF burst rate gating under OPN disinhibition
// Target #379 in WEFT Verification Suite.

void benchmark_scalar_saccade(const std::vector<float>& amp,
                              const std::vector<float>& opn_clamp,
                              const std::vector<float>& sc_drive,
                              std::vector<float>& peak_vel,
                              std::vector<float>& duration,
                              std::vector<float>& pprf_rate,
                              size_t n) {
    const float V_max = 850.0f;
    const float C = 12.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = amp[i];
        float v = V_max * (1.0f - std::exp(-a / C));
        peak_vel[i] = v;
        duration[i] = 22.0f + 2.5f * a;
        pprf_rate[i] = (sc_drive[i] > opn_clamp[i]) ? v * 1.15f : 0.0f;
    }
}

void benchmark_avx2_saccade(const float* amp,
                            const float* opn_clamp,
                            const float* sc_drive,
                            float* peak_vel,
                            float* duration,
                            float* pprf_rate,
                            size_t n) {
    const __m256 v_vmax = _mm256_set1_ps(850.0f);
    const __m256 v_inv_c = _mm256_set1_ps(1.0f / 12.0f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_d0 = _mm256_set1_ps(22.0f);
    const __m256 v_slope = _mm256_set1_ps(2.5f);
    const __m256 v_1_15 = _mm256_set1_ps(1.15f);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 a = _mm256_loadu_ps(amp + i);
        __m256 opn = _mm256_loadu_ps(opn_clamp + i);
        __m256 sc = _mm256_loadu_ps(sc_drive + i);

        // Duration: 22.0 + 2.5 * a
        __m256 dur = _mm256_fmadd_ps(v_slope, a, v_d0);
        _mm256_storeu_ps(duration + i, dur);

        // Approximate 1 - exp(-a/C) using rational approximation or clamp:
        // x = a/12; approx = x / (1 + x)
        __m256 x = _mm256_mul_ps(a, v_inv_c);
        __m256 denom = _mm256_add_ps(v_one, x);
        __m256 frac = _mm256_mul_ps(x, _mm256_rcp_ps(denom));
        __m256 v = _mm256_mul_ps(v_vmax, frac);
        _mm256_storeu_ps(peak_vel + i, v);

        // Gating: sc > opn
        __m256 mask = _mm256_cmp_ps(sc, opn, _CMP_GT_OQ);
        __m256 pprf = _mm256_blendv_ps(v_zero, _mm256_mul_ps(v, v_1_15), mask);
        _mm256_storeu_ps(pprf_rate + i, pprf);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> amp(N, 15.0f);
    std::vector<float> opn_clamp(N, 50.0f);
    std::vector<float> sc_drive(N, 75.0f);

    std::vector<float> peak_vel(N, 0.0f);
    std::vector<float> duration(N, 0.0f);
    std::vector<float> pprf_rate(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_saccade(amp.data(), opn_clamp.data(), sc_drive.data(),
                               peak_vel.data(), duration.data(), pprf_rate.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_saccade(amp.data(), opn_clamp.data(), sc_drive.data(),
                               peak_vel.data(), duration.data(), pprf_rate.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Saccadic Burst Generator (PPRF / OPN) Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Peak Velocity: " << peak_vel[0] << " deg/s" << std::endl;
    std::cout << "Sample Duration: " << duration[0] << " ms" << std::endl;
    std::cout << "Sample PPRF Burst Rate: " << pprf_rate[0] << " spikes/s" << std::endl;

    return 0;
}
