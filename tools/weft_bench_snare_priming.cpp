// tools/weft_bench_snare_priming.cpp
// Native AVX2/FMA SIMD benchmark for Presynaptic SNARE Complex Assembly & Munc18-1 Priming
// Computes open syntaxin availability, core SNARE zippering, RRP vesicle priming, and quantal release jitter

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr size_t N_CHANNELS = 65536;
static constexpr int N_ITERATIONS = 1000;

void benchmark_snare_priming_avx2(
    const float* __restrict__ munc13_drive,
    const float* __restrict__ ca_microdomain,
    float* __restrict__ out_open_syn,
    float* __restrict__ out_snare_zip,
    float* __restrict__ out_rrp,
    float* __restrict__ out_jitter,
    size_t n
) {
    const __m256 v_munc13_scale = _mm256_set1_ps(0.98f);
    const __m256 v_max_open = _mm256_set1_ps(98.0f);
    const __m256 v_hundred = _mm256_set1_ps(100.0f);
    const __m256 v_k_zip = _mm256_set1_ps(10.0f);
    const __m256 v_rrp_scale = _mm256_set1_ps(0.95f);
    const __m256 v_base_num = _mm256_set1_ps(1800.0f);
    const __m256 v_penalty_mult = _mm256_set1_ps(12.0f);
    const __m256 v_eighty = _mm256_set1_ps(80.0f);
    const __m256 v_min_jitter = _mm256_set1_ps(120.0f);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_m13 = _mm256_loadu_ps(munc13_drive + i);
        __m256 v_ca = _mm256_loadu_ps(ca_microdomain + i);

        // open_syn = min(98.0, m13 * 0.98)
        __m256 v_open = _mm256_min_ps(v_max_open, _mm256_mul_ps(v_m13, v_munc13_scale));
        _mm256_storeu_ps(out_open_syn + i, v_open);

        // zip = (open / (open + 10.0)) * 100.0
        __m256 v_denom = _mm256_add_ps(v_open, v_k_zip);
        __m256 v_zip = _mm256_mul_ps(_mm256_div_ps(v_open, v_denom), v_hundred);
        _mm256_storeu_ps(out_snare_zip + i, v_zip);

        // rrp = zip * 0.95
        __m256 v_rrp = _mm256_mul_ps(v_zip, v_rrp_scale);
        _mm256_storeu_ps(out_rrp + i, v_rrp);

        // jitter = max(120.0, 1800.0 / ca + max(0.0, 80.0 - rrp) * 12.0)
        __m256 v_base_jit = _mm256_div_ps(v_base_num, v_ca);
        __m256 v_diff = _mm256_sub_ps(v_eighty, v_rrp);
        __m256 v_deficit = _mm256_max_ps(v_zero, v_diff);
        __m256 v_penalty = _mm256_mul_ps(v_deficit, v_penalty_mult);
        __m256 v_jit_raw = _mm256_add_ps(v_base_jit, v_penalty);
        __m256 v_jit = _mm256_max_ps(v_min_jitter, v_jit_raw);
        _mm256_storeu_ps(out_jitter + i, v_jit);
    }
}

int main() {
    std::vector<float> m13(N_CHANNELS, 85.0f);
    std::vector<float> ca(N_CHANNELS, 15.0f);
    std::vector<float> open_syn(N_CHANNELS);
    std::vector<float> zip(N_CHANNELS);
    std::vector<float> rrp(N_CHANNELS);
    std::vector<float> jitter(N_CHANNELS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        benchmark_snare_priming_avx2(
            m13.data(), ca.data(),
            open_syn.data(), zip.data(), rrp.data(), jitter.data(),
            N_CHANNELS
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERATIONS; ++i) {
        benchmark_snare_priming_avx2(
            m13.data(), ca.data(),
            open_syn.data(), zip.data(), rrp.data(), jitter.data(),
            N_CHANNELS
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / N_ITERATIONS;
    double throughput_mch_s = (static_cast<double>(N_CHANNELS) / (avg_us * 1e-6)) / 1e6;

    std::cout << "[AVX2-FMA Benchmark] Presynaptic SNARE Priming Kinetics" << std::endl;
    std::cout << "Channels: " << N_CHANNELS << " | Iterations: " << N_ITERATIONS << std::endl;
    std::cout << "Avg latency per iteration: " << avg_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput_mch_s << " Million channels/sec" << std::endl;
    std::cout << "Sample verification [Channel 0]:" << std::endl;
    std::cout << "  Munc13 Drive: " << m13[0] << "% | Ca2+ Microdomain: " << ca[0] << " uM" << std::endl;
    std::cout << "  Open Syntaxin: " << open_syn[0] << "% | SNARE Zippering: " << zip[0] << "%" << std::endl;
    std::cout << "  RRP Occupancy: " << rrp[0] << "% | Release Jitter: " << jitter[0] << " us" << std::endl;

    return 0;
}
