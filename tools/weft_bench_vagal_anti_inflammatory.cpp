// tools/weft_bench_vagal_anti_inflammatory.cpp
// Native AVX2/FMA SIMD benchmark for Vagal Cholinergic Anti-Inflammatory Reflex kinetics
// Computes parallel acetylcholine binding to alpha7nAChR, STAT3 phosphorylation, and NF-kB suppression

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr size_t N_CHANNELS = 65536;
static constexpr int N_ITERATIONS = 1000;

void benchmark_vagal_anti_inflammatory_avx2(
    const float* __restrict__ vagus_freq,
    const float* __restrict__ lps_insult,
    float* __restrict__ out_ach,
    float* __restrict__ out_alpha7_occ,
    float* __restrict__ out_stat3,
    float* __restrict__ out_nfkb,
    float* __restrict__ out_tnf_alpha,
    size_t n
) {
    const __m256 v_ach_gain = _mm256_set1_ps(24.5f);
    const __m256 v_ach_max = _mm256_set1_ps(600.0f);
    const __m256 v_k_ach = _mm256_set1_ps(120.0f);
    const __m256 v_hundred = _mm256_set1_ps(100.0f);
    const __m256 v_stat3_gain = _mm256_set1_ps(0.045f);
    const __m256 v_suppr_gain = _mm256_set1_ps(0.82f);
    const __m256 v_nfkb_min = _mm256_set1_ps(8.0f);
    const __m256 v_tnf_base = _mm256_set1_ps(15.0f);
    const __m256 v_tnf_slope = _mm256_set1_ps(1.4f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_freq = _mm256_loadu_ps(vagus_freq + i);
        __m256 v_lps = _mm256_loadu_ps(lps_insult + i);

        // Splenic ACh concentration: min(600, freq * 24.5)
        __m256 v_ach = _mm256_min_ps(v_ach_max, _mm256_mul_ps(v_freq, v_ach_gain));
        _mm256_storeu_ps(out_ach + i, v_ach);

        // alpha7nAChR occupancy: (ach / (ach + 120)) * 100
        __m256 v_denom = _mm256_add_ps(v_ach, v_k_ach);
        __m256 v_occ = _mm256_mul_ps(_mm256_div_ps(v_ach, v_denom), v_hundred);
        _mm256_storeu_ps(out_alpha7_occ + i, v_occ);

        // STAT3 phosphorylation ratio: occ * 0.045
        __m256 v_stat3 = _mm256_mul_ps(v_occ, v_stat3_gain);
        _mm256_storeu_ps(out_stat3 + i, v_stat3);

        // NF-kB translocation: max(8.0, lps - occ * 0.82)
        __m256 v_suppr = _mm256_mul_ps(v_occ, v_suppr_gain);
        __m256 v_nfkb_raw = _mm256_sub_ps(v_lps, v_suppr);
        __m256 v_nfkb = _mm256_max_ps(v_nfkb_min, _mm256_min_ps(v_hundred, v_nfkb_raw));
        _mm256_storeu_ps(out_nfkb + i, v_nfkb);

        // Plasma TNF-alpha: 15.0 + nfkb * 1.4
        __m256 v_tnf = _mm256_fmadd_ps(v_nfkb, v_tnf_slope, v_tnf_base);
        _mm256_storeu_ps(out_tnf_alpha + i, v_tnf);
    }
}

int main() {
    std::vector<float> freq(N_CHANNELS, 20.0f);
    std::vector<float> lps(N_CHANNELS, 80.0f);
    std::vector<float> ach(N_CHANNELS);
    std::vector<float> occ(N_CHANNELS);
    std::vector<float> stat3(N_CHANNELS);
    std::vector<float> nfkb(N_CHANNELS);
    std::vector<float> tnf(N_CHANNELS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        benchmark_vagal_anti_inflammatory_avx2(
            freq.data(), lps.data(),
            ach.data(), occ.data(), stat3.data(), nfkb.data(), tnf.data(),
            N_CHANNELS
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERATIONS; ++i) {
        benchmark_vagal_anti_inflammatory_avx2(
            freq.data(), lps.data(),
            ach.data(), occ.data(), stat3.data(), nfkb.data(), tnf.data(),
            N_CHANNELS
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / N_ITERATIONS;
    double throughput_mch_s = (static_cast<double>(N_CHANNELS) / (avg_us * 1e-6)) / 1e6;

    std::cout << "[AVX2-FMA Benchmark] Vagal Cholinergic Anti-Inflammatory Kinetics" << std::endl;
    std::cout << "Channels: " << N_CHANNELS << " | Iterations: " << N_ITERATIONS << std::endl;
    std::cout << "Avg latency per iteration: " << avg_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput_mch_s << " Million channels/sec" << std::endl;
    std::cout << "Sample verification [Channel 0]:" << std::endl;
    std::cout << "  ACh: " << ach[0] << " nM | alpha7 Occupancy: " << occ[0] << "%" << std::endl;
    std::cout << "  pSTAT3: " << stat3[0] << "x | NF-kB: " << nfkb[0] << "% | TNF-alpha: " << tnf[0] << " pg/mL" << std::endl;

    return 0;
}
