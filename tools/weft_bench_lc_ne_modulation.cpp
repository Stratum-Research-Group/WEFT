// Vectorized microbenchmark for Locus Coeruleus Norepinephrine (LC-NE) Cortical Gain Modulation
// Evaluates AVX2 + FMA SIMD throughput of neuromodulatory gain scaling, noise injection, and SNR calculation.
// Part of the WEFT Monorepo benchmark suite (Target #386).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) LCNECorticalBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> tonic_rate;
    std::vector<float> phasic_burst;
    std::vector<float> input_salience;
    std::vector<float> baseline_noise;
    std::vector<float> output_gain;
    std::vector<float> output_signal;
    std::vector<float> output_noise;
    std::vector<float> output_snr;

    LCNECorticalBatch()
        : tonic_rate(N), phasic_burst(N), input_salience(N), baseline_noise(N),
          output_gain(N), output_signal(N), output_noise(N), output_snr(N) {
        for (size_t i = 0; i < N; ++i) {
            tonic_rate[i] = 0.5f + (static_cast<float>(i % 100) / 10.0f); // 0.5 to 10.5 Hz
            phasic_burst[i] = (i % 3 == 0) ? 4.5f : 0.0f; // Salient bursts
            input_salience[i] = 1.0f + (static_cast<float>(i % 50) / 10.0f);
            baseline_noise[i] = 1.5f;
        }
    }
};

void benchmark_lc_ne_avx2(LCNECorticalBatch& batch) {
    const size_t n = LCNECorticalBatch::N;
    const float* __restrict tonic = batch.tonic_rate.data();
    const float* __restrict phasic = batch.phasic_burst.data();
    const float* __restrict salience = batch.input_salience.data();
    const float* __restrict base_noise = batch.baseline_noise.data();
    float* __restrict out_gain = batch.output_gain.data();
    float* __restrict out_signal = batch.output_signal.data();
    float* __restrict out_noise = batch.output_noise.data();
    float* __restrict out_snr = batch.output_snr.data();

    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_alpha = _mm256_set1_ps(0.35f);
    const __m256 v_beta_noise = _mm256_set1_ps(0.20f);
    const __m256 v_min_snr = _mm256_set1_ps(0.01f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_tonic = _mm256_loadu_ps(tonic + i);
        __m256 v_phasic = _mm256_loadu_ps(phasic + i);
        __m256 v_salience = _mm256_loadu_ps(salience + i);
        __m256 v_bnoise = _mm256_loadu_ps(base_noise + i);

        // Effective NE = tonic + phasic
        __m256 v_eff_ne = _mm256_add_ps(v_tonic, v_phasic);

        // Gain G = 1.0 + alpha * effective_ne
        __m256 v_gain = _mm256_fmadd_ps(v_alpha, v_eff_ne, v_one);

        // Signal S = G * salience
        __m256 v_signal = _mm256_mul_ps(v_gain, v_salience);

        // Noise N = base_noise + beta_noise * tonic
        __m256 v_noise = _mm256_fmadd_ps(v_beta_noise, v_tonic, v_bnoise);

        // SNR = max(0.01, signal / noise)
        __m256 v_snr_raw = _mm256_div_ps(v_signal, v_noise);
        __m256 v_snr = _mm256_max_ps(v_min_snr, v_snr_raw);

        _mm256_storeu_ps(out_gain + i, v_gain);
        _mm256_storeu_ps(out_signal + i, v_signal);
        _mm256_storeu_ps(out_noise + i, v_noise);
        _mm256_storeu_ps(out_snr + i, v_snr);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA LC-NE Neuromodulatory Gain Microbenchmark\n";
    std::cout << "========================================================\n";

    LCNECorticalBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_lc_ne_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_lc_ne_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(LCNECorticalBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << LCNECorticalBatch::N << " elements\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (Tonic=" << batch.tonic_rate[0] 
              << ", Phasic=" << batch.phasic_burst[0] << "):\n";
    std::cout << "  Cortical Gain:  " << batch.output_gain[0] << "x\n";
    std::cout << "  Cortical Noise: " << batch.output_noise[0] << "\n";
    std::cout << "  Effective SNR:  " << batch.output_snr[0] << "\n";
    std::cout << "========================================================\n";

    return 0;
}
