// Vectorized microbenchmark for Basal Forebrain Acetylcholine (BF-ACh) Attention Cue Gating
// Evaluates AVX2 + FMA SIMD throughput of cholinergic feedforward amplification vs recurrent suppression.
// Part of the WEFT Monorepo benchmark suite (Target #388).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) AChCorticalBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> ach_tone;
    std::vector<float> phasic_surge;
    std::vector<float> feedforward_input;
    std::vector<float> recurrent_input;
    std::vector<float> output_ff_gain;
    std::vector<float> output_rec_weight;
    std::vector<float> output_cortical_response;
    std::vector<float> output_cue_detection_index;

    AChCorticalBatch()
        : ach_tone(N), phasic_surge(N), feedforward_input(N), recurrent_input(N),
          output_ff_gain(N), output_rec_weight(N), output_cortical_response(N), output_cue_detection_index(N) {
        for (size_t i = 0; i < N; ++i) {
            ach_tone[i] = 1.0f + (static_cast<float>(i % 100) / 10.0f); // 1.0 to 11.0 uM
            phasic_surge[i] = (i % 4 == 0) ? 2.5f : 0.0f; // Phasic cue surge
            feedforward_input[i] = 2.0f + (static_cast<float>(i % 50) / 10.0f);
            recurrent_input[i] = 4.0f;
        }
    }
};

void benchmark_ach_gating_avx2(AChCorticalBatch& batch) {
    const size_t n = AChCorticalBatch::N;
    const float* __restrict tone = batch.ach_tone.data();
    const float* __restrict surge = batch.phasic_surge.data();
    const float* __restrict in_ff = batch.feedforward_input.data();
    const float* __restrict in_rec = batch.recurrent_input.data();
    float* __restrict out_g_ff = batch.output_ff_gain.data();
    float* __restrict out_w_rec = batch.output_rec_weight.data();
    float* __restrict out_resp = batch.output_cortical_response.data();
    float* __restrict out_idx = batch.output_cue_detection_index.data();

    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_alpha = _mm256_set1_ps(0.35f);
    const __m256 v_beta = _mm256_set1_ps(0.50f);
    const __m256 v_noise_base = _mm256_set1_ps(1.5f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_tone = _mm256_loadu_ps(tone + i);
        __m256 v_surge = _mm256_loadu_ps(surge + i);
        __m256 v_in_ff = _mm256_loadu_ps(in_ff + i);
        __m256 v_in_rec = _mm256_loadu_ps(in_rec + i);

        // Effective ACh = tone + surge
        __m256 v_eff_ach = _mm256_add_ps(v_tone, v_surge);

        // Feedforward Gain: G_ff = 1.0 + 0.35 * eff_ach
        __m256 v_g_ff = _mm256_fmadd_ps(v_alpha, v_eff_ach, v_one);

        // Recurrent Weight: W_rec = 1.0 / (1.0 + 0.50 * eff_ach)
        __m256 v_w_rec_denom = _mm256_fmadd_ps(v_beta, v_eff_ach, v_one);
        __m256 v_w_rec = _mm256_div_ps(v_one, v_w_rec_denom);

        // Cortical Response: Resp = G_ff * in_ff + W_rec * in_rec
        __m256 v_sig_ff = _mm256_mul_ps(v_g_ff, v_in_ff);
        __m256 v_resp = _mm256_fmadd_ps(v_w_rec, v_in_rec, v_sig_ff);

        // Cue Detection Index = (G_ff * in_ff) / (1.5 + W_rec * in_rec)
        __m256 v_denom_cue = _mm256_fmadd_ps(v_w_rec, v_in_rec, v_noise_base);
        __m256 v_idx = _mm256_div_ps(v_sig_ff, v_denom_cue);

        _mm256_storeu_ps(out_g_ff + i, v_g_ff);
        _mm256_storeu_ps(out_w_rec + i, v_w_rec);
        _mm256_storeu_ps(out_resp + i, v_resp);
        _mm256_storeu_ps(out_idx + i, v_idx);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA Basal Forebrain ACh Attention Benchmark\n";
    std::cout << "========================================================\n";

    AChCorticalBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_ach_gating_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_ach_gating_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(AChCorticalBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << AChCorticalBatch::N << " elements\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (ACh=" << batch.ach_tone[0] 
              << " uM, Phasic=" << batch.phasic_surge[0] << " uM):\n";
    std::cout << "  Feedforward Gain (G_ff):  " << batch.output_ff_gain[0] << "x\n";
    std::cout << "  Recurrent Weight (W_rec): " << batch.output_rec_weight[0] << "\n";
    std::cout << "  Cortical Response:        " << batch.output_cortical_response[0] << "\n";
    std::cout << "  Cue Detection Index:      " << batch.output_cue_detection_index[0] << "\n";
    std::cout << "========================================================\n";

    return 0;
}
