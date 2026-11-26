// Vectorized microbenchmark for VTA Dopaminergic Three-Factor Synaptic Plasticity
// Evaluates AVX2 + FMA SIMD throughput of eligibility trace calculation, dopamine RPE gating, and synaptic weight consolidation.
// Part of the WEFT Monorepo benchmark suite (Target #390).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) PlasticityBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> pre_rate;
    std::vector<float> post_rate;
    std::vector<float> dopamine_rpe;
    std::vector<float> current_weight;
    std::vector<float> output_eligibility;
    std::vector<float> output_delta_w;
    std::vector<float> output_updated_weight;

    PlasticityBatch()
        : pre_rate(N), post_rate(N), dopamine_rpe(N), current_weight(N),
          output_eligibility(N), output_delta_w(N), output_updated_weight(N) {
        for (size_t i = 0; i < N; ++i) {
            pre_rate[i] = 5.0f + (static_cast<float>(i % 45)); // 5 - 50 Hz
            post_rate[i] = 2.0f + (static_cast<float>(i % 35)); // 2 - 37 Hz
            dopamine_rpe[i] = -1.5f + (static_cast<float>(i % 50) / 10.0f); // -1.5 to +3.5 RPE
            current_weight[i] = 1.0f + (static_cast<float>(i % 20) / 10.0f); // 1.0 - 3.0 uS
        }
    }
};

void benchmark_plasticity_avx2(PlasticityBatch& batch) {
    const size_t n = PlasticityBatch::N;
    const float* __restrict pre = batch.pre_rate.data();
    const float* __restrict post = batch.post_rate.data();
    const float* __restrict rpe = batch.dopamine_rpe.data();
    const float* __restrict cur_w = batch.current_weight.data();
    float* __restrict out_e = batch.output_eligibility.data();
    float* __restrict out_dw = batch.output_delta_w.data();
    float* __restrict out_w = batch.output_updated_weight.data();

    const __m256 v_scale_trace = _mm256_set1_ps(0.01f); // 1/100
    const __m256 v_eta = _mm256_set1_ps(0.08f);
    const __m256 v_min_w = _mm256_set1_ps(0.05f);
    const __m256 v_max_w = _mm256_set1_ps(3.50f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_pre = _mm256_loadu_ps(pre + i);
        __m256 v_post = _mm256_loadu_ps(post + i);
        __m256 v_rpe = _mm256_loadu_ps(rpe + i);
        __m256 v_w = _mm256_loadu_ps(cur_w + i);

        // Eligibility trace e = (pre * post) * 0.01
        __m256 v_trace_raw = _mm256_mul_ps(v_pre, v_post);
        __m256 v_trace = _mm256_mul_ps(v_trace_raw, v_scale_trace);

        // Synaptic delta dW = eta * trace * RPE
        __m256 v_dw = _mm256_mul_ps(v_eta, _mm256_mul_ps(v_trace, v_rpe));

        // Updated weight W_next = clamp(W + dW, min_w, max_w)
        __m256 v_w_raw = _mm256_add_ps(v_w, v_dw);
        __m256 v_w_clamped = _mm256_max_ps(v_min_w, _mm256_min_ps(v_max_w, v_w_raw));

        _mm256_storeu_ps(out_e + i, v_trace);
        _mm256_storeu_ps(out_dw + i, v_dw);
        _mm256_storeu_ps(out_w + i, v_w_clamped);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA VTA Three-Factor Plasticity Benchmark\n";
    std::cout << "========================================================\n";

    PlasticityBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_plasticity_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_plasticity_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(PlasticityBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << PlasticityBatch::N << " elements\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (Pre=" << batch.pre_rate[0] 
              << " Hz, Post=" << batch.post_rate[0] << " Hz, RPE=" << batch.dopamine_rpe[0] << "):\n";
    std::cout << "  Eligibility Trace: " << batch.output_eligibility[0] << "\n";
    std::cout << "  Synaptic Delta:    " << batch.output_delta_w[0] << "\n";
    std::cout << "  Updated Weight:    " << batch.output_updated_weight[0] << " uS\n";
    std::cout << "========================================================\n";

    return 0;
}
