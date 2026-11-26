#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Cerebellar Flocculus Purkinje Cell Plasticity & VOR Gain Adaptation
// Evaluates:
// 1. Retinal slip error computation: slip = |target_gain - current_gain| * head_speed
// 2. Climbing fiber-triggered PF-Purkinje LTD synaptic weight decrement:
//    dw = eta * slip
// 3. Compensatory VOR gain increment
// Target #378 in WEFT Verification Suite.

void benchmark_scalar_cerebellar(const std::vector<float>& target_gain,
                                 std::vector<float>& current_gain,
                                 std::vector<float>& purkinje_weight,
                                 std::vector<float>& retinal_slip,
                                 size_t n) {
    const float eta = 0.005f;
    for (size_t i = 0; i < n; ++i) {
        float tg = target_gain[i];
        float cg = current_gain[i];
        float slip = std::fabs(tg - cg) * 100.0f;
        retinal_slip[i] = slip;

        // LTD: reduce Purkinje weight to increase vestibular disinhibition
        float dw = eta * slip;
        purkinje_weight[i] = std::fmax(0.0f, purkinje_weight[i] - dw);
        current_gain[i] = cg + 0.001f * dw;
    }
}

void benchmark_avx2_cerebellar(const float* target_gain,
                               float* current_gain,
                               float* purkinje_weight,
                               float* retinal_slip,
                               size_t n) {
    const __m256 v_100 = _mm256_set1_ps(100.0f);
    const __m256 v_eta = _mm256_set1_ps(0.005f);
    const __m256 v_gain_rate = _mm256_set1_ps(0.001f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for (size_t i = 0; i < n; i += 8) {
        __m256 tg = _mm256_loadu_ps(target_gain + i);
        __m256 cg = _mm256_loadu_ps(current_gain + i);
        __m256 pw = _mm256_loadu_ps(purkinje_weight + i);

        // Retinal slip: |tg - cg| * 100.0
        __m256 diff = _mm256_sub_ps(tg, cg);
        __m256 abs_diff = _mm256_and_ps(diff, v_sign_mask);
        __m256 slip = _mm256_mul_ps(abs_diff, v_100);
        _mm256_storeu_ps(retinal_slip + i, slip);

        // Plasticity step
        __m256 dw = _mm256_mul_ps(v_eta, slip);
        __m256 new_pw = _mm256_max_ps(_mm256_sub_ps(pw, dw), v_zero);
        __m256 new_cg = _mm256_fmadd_ps(v_gain_rate, dw, cg);

        _mm256_storeu_ps(purkinje_weight + i, new_pw);
        _mm256_storeu_ps(current_gain + i, new_cg);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> target_gain(N, 1.5f);
    std::vector<float> current_gain(N, 1.0f);
    std::vector<float> purkinje_weight(N, 1.0f);
    std::vector<float> retinal_slip(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_cerebellar(target_gain.data(), current_gain.data(),
                                  purkinje_weight.data(), retinal_slip.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_cerebellar(target_gain.data(), current_gain.data(),
                                  purkinje_weight.data(), retinal_slip.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Cerebellar Floccular Plasticity Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Adapted Gain: " << current_gain[0] << std::endl;
    std::cout << "Sample Purkinje Synaptic Weight: " << purkinje_weight[0] << std::endl;
    std::cout << "Sample Retinal Slip: " << retinal_slip[0] << " deg/s" << std::endl;

    return 0;
}
