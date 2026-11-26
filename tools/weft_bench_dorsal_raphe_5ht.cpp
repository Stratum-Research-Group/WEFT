// Vectorized microbenchmark for Dorsal Raphe Serotonin (5-HT) Temporal Patience and Discounting
// Evaluates AVX2 + FMA SIMD throughput of hyperbolic temporal discounting across delayed reward trials.
// Part of the WEFT Monorepo benchmark suite (Target #387).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) SerotoninDiscountBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> serotonin_tone;
    std::vector<float> delay_duration;
    std::vector<float> nominal_reward;
    std::vector<float> opportunity_cost;
    std::vector<float> output_effective_k;
    std::vector<float> output_discounted_value;
    std::vector<float> output_net_value;
    std::vector<float> output_perseverance_prob;

    SerotoninDiscountBatch()
        : serotonin_tone(N), delay_duration(N), nominal_reward(N), opportunity_cost(N),
          output_effective_k(N), output_discounted_value(N), output_net_value(N), output_perseverance_prob(N) {
        for (size_t i = 0; i < N; ++i) {
            serotonin_tone[i] = 0.2f + (static_cast<float>(i % 100) / 10.0f); // 0.2 to 10.2 Hz
            delay_duration[i] = 0.5f + (static_cast<float>(i % 50) / 10.0f);  // 0.5 to 5.5 s
            nominal_reward[i] = 50.0f;
            opportunity_cost[i] = 1.0f;
        }
    }
};

void benchmark_serotonin_patience_avx2(SerotoninDiscountBatch& batch) {
    const size_t n = SerotoninDiscountBatch::N;
    const float* __restrict tone = batch.serotonin_tone.data();
    const float* __restrict delay = batch.delay_duration.data();
    const float* __restrict reward = batch.nominal_reward.data();
    const float* __restrict cost = batch.opportunity_cost.data();
    float* __restrict out_k = batch.output_effective_k.data();
    float* __restrict out_val = batch.output_discounted_value.data();
    float* __restrict out_net = batch.output_net_value.data();
    float* __restrict out_prob = batch.output_perseverance_prob.data();

    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_kbase = _mm256_set1_ps(0.85f);
    const __m256 v_gamma = _mm256_set1_ps(0.45f);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_tone = _mm256_loadu_ps(tone + i);
        __m256 v_delay = _mm256_loadu_ps(delay + i);
        __m256 v_reward = _mm256_loadu_ps(reward + i);
        __m256 v_cost = _mm256_loadu_ps(cost + i);

        // Effective discount factor: k_eff = k_base / (1.0 + gamma * 5HT_tone)
        __m256 v_k_denom = _mm256_fmadd_ps(v_gamma, v_tone, v_one);
        __m256 v_k_eff = _mm256_div_ps(v_kbase, v_k_denom);

        // Hyperbolic Value: V(t) = R / (1.0 + k_eff * delay)
        __m256 v_v_denom = _mm256_fmadd_ps(v_k_eff, v_delay, v_one);
        __m256 v_val = _mm256_div_ps(v_reward, v_v_denom);

        // Net subjective value: Net_V = V(t) - cost * delay
        __m256 v_net = _mm256_fmsub_ps(v_cost, _mm256_sub_ps(v_zero, v_delay), v_val);
        // Correcting: Net_V = v_val - v_cost * v_delay
        __m256 v_cost_acc = _mm256_mul_ps(v_cost, v_delay);
        v_net = _mm256_sub_ps(v_val, v_cost_acc);

        // Perseverance probability proxy (clipped linear approximation of sigmoid):
        // p = clamp(0.5 + 0.02 * net, 0.0, 1.0)
        __m256 v_prob_raw = _mm256_fmadd_ps(_mm256_set1_ps(0.02f), v_net, _mm256_set1_ps(0.50f));
        __m256 v_prob = _mm256_max_ps(v_zero, _mm256_min_ps(v_one, v_prob_raw));

        _mm256_storeu_ps(out_k + i, v_k_eff);
        _mm256_storeu_ps(out_val + i, v_val);
        _mm256_storeu_ps(out_net + i, v_net);
        _mm256_storeu_ps(out_prob + i, v_prob);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA Dorsal Raphe 5-HT Patience Microbenchmark\n";
    std::cout << "========================================================\n";

    SerotoninDiscountBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_serotonin_patience_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_serotonin_patience_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(SerotoninDiscountBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << SerotoninDiscountBatch::N << " elements\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (5-HT=" << batch.serotonin_tone[0] 
              << " Hz, Delay=" << batch.delay_duration[0] << " s):\n";
    std::cout << "  Effective k:       " << batch.output_effective_k[0] << " /s\n";
    std::cout << "  Discounted Value:  " << batch.output_discounted_value[0] << "\n";
    std::cout << "  Net Value:         " << batch.output_net_value[0] << "\n";
    std::cout << "  Perseverance Prob: " << (batch.output_perseverance_prob[0] * 100.0f) << "%\n";
    std::cout << "========================================================\n";

    return 0;
}
