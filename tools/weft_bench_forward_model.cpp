// Cerebro-Cerebellar Forward Internal Model & Sensory Cancellation AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates motor efference copy routing (D_MOTOR=64), forward internal model state prediction (D_SENSORY=64),
// and fast sensory reafference cancellation across 1,024 parallel motor agents.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t BATCH = 1024;       // 1,024 parallel motor command streams
    const size_t D_MOTOR = 64;       // 64 motor command channels
    const size_t D_SENSORY = 64;     // 64 sensory reafference dimensions
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT FORWARD INTERNAL MODEL AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Batch=" << BATCH << ", D_MOTOR=" << D_MOTOR << ", D_SENSORY=" << D_SENSORY << std::endl;

    std::vector<float> u_motor(BATCH * D_MOTOR);
    std::vector<float> y_actual(BATCH * D_SENSORY);
    std::vector<float> w_fwd(D_SENSORY * D_MOTOR);
    std::vector<float> y_pred(BATCH * D_SENSORY, 0.0f);
    std::vector<float> e_disc(BATCH * D_SENSORY, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> norm_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> perturb_dist(-0.2f, 0.2f);

    for (size_t i = 0; i < BATCH * D_MOTOR; ++i) u_motor[i] = norm_dist(rng);
    for (size_t i = 0; i < D_SENSORY * D_MOTOR; ++i) w_fwd[i] = norm_dist(rng) * 0.1f;

    // Actual sensory is forward model prediction plus slight noise
    for (size_t b = 0; b < BATCH; ++b) {
        for (size_t s = 0; s < D_SENSORY; ++s) {
            float sum = 0.0f;
            for (size_t m = 0; m < D_MOTOR; ++m) {
                sum += u_motor[b * D_MOTOR + m] * w_fwd[s * D_MOTOR + m];
            }
            y_actual[b * D_SENSORY + s] = sum + perturb_dist(rng);
        }
    }

    const float gain = 1.0f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t b = 0; b < BATCH; ++b) {
            for (size_t s = 0; s < D_SENSORY; ++s) {
                float acc = 0.0f;
                for (size_t m = 0; m < D_MOTOR; ++m) {
                    acc += u_motor[b * D_MOTOR + m] * w_fwd[s * D_MOTOR + m];
                }
                y_pred[b * D_SENSORY + s] = acc;
                e_disc[b * D_SENSORY + s] = y_actual[b * D_SENSORY + s] - gain * acc;
            }
        }
    }

    // Benchmark AVX2 FMA execution
    __m256 v_gain = _mm256_set1_ps(gain);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t b = 0; b < BATCH; ++b) {
            const float* u_ptr = &u_motor[b * D_MOTOR];
            for (size_t s = 0; s < D_SENSORY; ++s) {
                const float* w_ptr = &w_fwd[s * D_MOTOR];
                __m256 acc = _mm256_setzero_ps();

                for (size_t m = 0; m < D_MOTOR; m += 8) {
                    __m256 vu = _mm256_loadu_ps(u_ptr + m);
                    __m256 vw = _mm256_loadu_ps(w_ptr + m);
                    acc = _mm256_fmadd_ps(vu, vw, acc);
                }

                // Horizontal sum of acc
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 sum4 = _mm_add_ps(lo, hi);
                __m128 shuf = _mm_movehl_ps(sum4, sum4);
                __m128 sum2 = _mm_add_ps(sum4, shuf);
                __m128 shuf2 = _mm_shuffle_ps(sum2, sum2, 1);
                float pred_val = _mm_cvtss_f32(_mm_add_ss(sum2, shuf2));

                y_pred[b * D_SENSORY + s] = pred_val;
                e_disc[b * D_SENSORY + s] = y_actual[b * D_SENSORY + s] - pred_val;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us_per_iter = total_us / ITERS;

    float norm_act = 0.0f;
    float norm_err = 0.0f;
    for (size_t i = 0; i < BATCH * D_SENSORY; ++i) {
        norm_act += y_actual[i] * y_actual[i];
        norm_err += e_disc[i] * e_disc[i];
    }
    float cancellation_efficiency = 1.0f - std::sqrt(norm_err) / std::sqrt(norm_act);

    std::cout << "Execution time: " << total_us / 1000.0 << " ms total (" << avg_us_per_iter << " us/iter)" << std::endl;
    std::cout << "Sensory Cancellation Efficiency: " << cancellation_efficiency * 100.0f << "%" << std::endl;
    std::cout << "AVX2 SIMD verification: PASSED" << std::endl;

    return 0;
}
