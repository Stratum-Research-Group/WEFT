#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: Supplementary Motor Area (SMA) Sequence Chunking
// Benchmarks 32,768 action chunk state machines.

void benchmark_sma_chunking_avx2(size_t n, size_t iterations) {
    std::vector<float> elapsed_ms(n);
    std::vector<float> target_dur_ms(n);
    std::vector<float> pre_sma_out(n, 0.0f);
    std::vector<float> sma_rate_out(n, 0.0f);
    std::vector<float> m1_gate_out(n, 0.0f);
    std::vector<float> sync_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        elapsed_ms[i] = 10.0f + 250.0f * (static_cast<float>(i) / static_cast<float>(n));
        target_dur_ms[i] = 200.0f;
    }

    const __m256 v_fbase = _mm256_set1_ps(25.0f);
    const __m256 v_kramp = _mm256_set1_ps(145.0f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_tau_thresh = _mm256_set1_ps(0.85f);
    const __m256 v_pre_scale = _mm256_set1_ps(180.0f / 0.15f);
    const __m256 v_pre_base = _mm256_set1_ps(10.0f);
    const __m256 v_gate_low = _mm256_set1_ps(0.10f);
    const __m256 v_gate_high = _mm256_set1_ps(0.98f);
    const __m256 v_sync_base = _mm256_set1_ps(0.95f);
    const __m256 v_sync_gain = _mm256_set1_ps(0.04f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 v_t = _mm256_loadu_ps(&elapsed_ms[i]);
            __m256 v_dur = _mm256_loadu_ps(&target_dur_ms[i]);

            __m256 tau = _mm256_div_ps(v_t, v_dur);
            __m256 tau_clamped = _mm256_min_ps(tau, v_one);

            // SMA timing: fbase + kramp * min(tau, 1.0)
            __m256 sma_rate = _mm256_add_ps(v_fbase, _mm256_mul_ps(v_kramp, tau_clamped));

            // Pre-SMA chunk switch drive: if tau >= 0.85, 180 * (tau - 0.85) / 0.15, else 10.0
            __m256 diff_tau = _mm256_sub_ps(tau, v_tau_thresh);
            __m256 pre_burst = _mm256_mul_ps(diff_tau, v_pre_scale);
            __m256 is_switch = _mm256_cmp_ps(tau, v_tau_thresh, _CMP_GE_OQ);
            __m256 pre_sma = _mm256_blendv_ps(v_pre_base, pre_burst, is_switch);

            // M1 gate: permissive (1.0) if tau between 0.10 and 0.98
            __m256 ge_low = _mm256_cmp_ps(tau, v_gate_low, _CMP_GE_OQ);
            __m256 le_high = _mm256_cmp_ps(tau, v_gate_high, _CMP_LE_OQ);
            __m256 is_permissive = _mm256_and_ps(ge_low, le_high);
            __m256 m1_gate = _mm256_blendv_ps(v_zero, v_one, is_permissive);

            // Bimanual sync: 0.95 + 0.04 * (1.0 - 0.5 * (tau - 0.5)^2 * 10.0 approx)
            __m256 sync_val = _mm256_add_ps(v_sync_base, _mm256_mul_ps(v_sync_gain, tau_clamped));

            _mm256_storeu_ps(&pre_sma_out[i], pre_sma);
            _mm256_storeu_ps(&sma_rate_out[i], sma_rate);
            _mm256_storeu_ps(&m1_gate_out[i], m1_gate);
            _mm256_storeu_ps(&sync_out[i], sync_val);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] SMA Action Chunk Units: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MChunks/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: t=" << elapsed_ms[n/2] << " ms => SMA=" 
              << sma_rate_out[n/2] << " Hz, Pre-SMA=" << pre_sma_out[n/2] << " Hz, M1Gate=" 
              << (m1_gate_out[n/2] > 0.5f ? "PERMISSIVE" : "LOCKED") << ", Sync=" << sync_out[n/2] * 100.0f << "%" << std::endl;
}

int main() {
    benchmark_sma_chunking_avx2(32768, 5000);
    return 0;
}
