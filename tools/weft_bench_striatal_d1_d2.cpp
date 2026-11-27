// tools/weft_bench_striatal_d1_d2.cpp
// Native AVX2/FMA SIMD benchmark for Striatal MSN D1/D2 Dopaminergic Plasticity Kinetics
// Computes D1 PKA phosphorylation, D2 Calcineurin modulation, Direct/Indirect pathway gains, and initiation ratio

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr size_t N_CHANNELS = 65536;
static constexpr int N_ITERATIONS = 1000;

void benchmark_striatal_d1_d2_avx2(
    const float* __restrict__ da_conc,
    const float* __restrict__ glut_drive,
    float* __restrict__ out_pka,
    float* __restrict__ out_cn,
    float* __restrict__ out_d1_gain,
    float* __restrict__ out_d2_gain,
    float* __restrict__ out_ratio,
    size_t n
) {
    const __m256 v_pka_base = _mm256_set1_ps(0.5f);
    const __m256 v_pka_amp = _mm256_set1_ps(4.5f);
    const __m256 v_k_d1 = _mm256_set1_ps(600.0f);
    const __m256 v_cn_base = _mm256_set1_ps(4.5f);
    const __m256 v_cn_amp = _mm256_set1_ps(3.5f);
    const __m256 v_k_d2 = _mm256_set1_ps(150.0f);
    const __m256 v_hundred = _mm256_set1_ps(100.0f);
    const __m256 v_d1_scale = _mm256_set1_ps(0.0075f);
    const __m256 v_d2_scale = _mm256_set1_ps(0.0045f);
    const __m256 v_eps = _mm256_set1_ps(0.01f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_da = _mm256_loadu_ps(da_conc + i);
        __m256 v_glu = _mm256_loadu_ps(glut_drive + i);

        // pka = 0.5 + (da / (da + 600.0)) * 4.5
        __m256 v_d1_denom = _mm256_add_ps(v_da, v_k_d1);
        __m256 v_pka_norm = _mm256_div_ps(v_da, v_d1_denom);
        __m256 v_pka = _mm256_fmadd_ps(v_pka_norm, v_pka_amp, v_pka_base);
        _mm256_storeu_ps(out_pka + i, v_pka);

        // cn = 4.5 - (da / (da + 150.0)) * 3.5
        __m256 v_d2_denom = _mm256_add_ps(v_da, v_k_d2);
        __m256 v_cn_norm = _mm256_div_ps(v_da, v_d2_denom);
        __m256 v_cn = _mm256_fnmadd_ps(v_cn_norm, v_cn_amp, v_cn_base);
        _mm256_storeu_ps(out_cn + i, v_cn);

        // d1_gain = (glu / 100.0) * pka * 0.75
        __m256 v_d1 = _mm256_mul_ps(_mm256_mul_ps(v_glu, v_pka), v_d1_scale);
        _mm256_storeu_ps(out_d1_gain + i, v_d1);

        // d2_gain = (glu / 100.0) * cn * 0.45
        __m256 v_d2 = _mm256_mul_ps(_mm256_mul_ps(v_glu, v_cn), v_d2_scale);
        _mm256_storeu_ps(out_d2_gain + i, v_d2);

        // ratio = d1 / max(0.01, d2)
        __m256 v_d2_safe = _mm256_max_ps(v_eps, v_d2);
        __m256 v_rat = _mm256_div_ps(v_d1, v_d2_safe);
        _mm256_storeu_ps(out_ratio + i, v_rat);
    }
}

int main() {
    std::vector<float> da(N_CHANNELS, 850.0f);
    std::vector<float> glut(N_CHANNELS, 75.0f);
    std::vector<float> pka(N_CHANNELS);
    std::vector<float> cn(N_CHANNELS);
    std::vector<float> d1(N_CHANNELS);
    std::vector<float> d2(N_CHANNELS);
    std::vector<float> ratio(N_CHANNELS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        benchmark_striatal_d1_d2_avx2(
            da.data(), glut.data(),
            pka.data(), cn.data(), d1.data(), d2.data(), ratio.data(),
            N_CHANNELS
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERATIONS; ++i) {
        benchmark_striatal_d1_d2_avx2(
            da.data(), glut.data(),
            pka.data(), cn.data(), d1.data(), d2.data(), ratio.data(),
            N_CHANNELS
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / N_ITERATIONS;
    double throughput_mch_s = (static_cast<double>(N_CHANNELS) / (avg_us * 1e-6)) / 1e6;

    std::cout << "[AVX2-FMA Benchmark] Striatal MSN D1/D2 Plasticity Kinetics" << std::endl;
    std::cout << "Channels: " << N_CHANNELS << " | Iterations: " << N_ITERATIONS << std::endl;
    std::cout << "Avg latency per iteration: " << avg_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput_mch_s << " Million channels/sec" << std::endl;
    std::cout << "Sample verification [Channel 0]:" << std::endl;
    std::cout << "  [DA]: " << da[0] << " nM | Glut Drive: " << glut[0] << "%" << std::endl;
    std::cout << "  D1 PKA: " << pka[0] << "x | D2 Calcineurin: " << cn[0] << "x" << std::endl;
    std::cout << "  Direct Gain: " << d1[0] << " | Indirect Gain: " << d2[0] << " | D1/D2 Ratio: " << ratio[0] << std::endl;

    return 0;
}
