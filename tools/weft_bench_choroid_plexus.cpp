// tools/weft_bench_choroid_plexus.cpp
// Native AVX2/FMA SIMD benchmark for Choroid Plexus CSF Secretion & Intracranial Pressure (ICP) Kinetics
// Computes CA-II catalysis, NKCC1 cotransport, osmotic driving gradient, CSF secretion, and compliance reserve

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr size_t N_CHANNELS = 65536;
static constexpr int N_ITERATIONS = 1000;

void benchmark_choroid_plexus_avx2(
    const float* __restrict__ ca2_drive,
    const float* __restrict__ nkcc1_rate,
    float* __restrict__ out_grad,
    float* __restrict__ out_csf,
    float* __restrict__ out_icp,
    float* __restrict__ out_comp,
    size_t n
) {
    const __m256 v_base_grad = _mm256_set1_ps(6.0f);
    const __m256 v_ca2_coeff = _mm256_set1_ps(0.105f);
    const __m256 v_nkcc1_coeff = _mm256_set1_ps(0.080f);
    const __m256 v_csf_scale = _mm256_set1_ps(0.36f / 20.0f);
    const __m256 v_icp_base = _mm256_set1_ps(7.0f);
    const __m256 v_icp_slope = _mm256_set1_ps(4.5f / 0.35f);
    const __m256 v_hundred = _mm256_set1_ps(100.0f);
    const __m256 v_comp_slope = _mm256_set1_ps(8.0f);
    const __m256 v_min_comp = _mm256_set1_ps(10.0f);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_ca = _mm256_loadu_ps(ca2_drive + i);
        __m256 v_nk = _mm256_loadu_ps(nkcc1_rate + i);

        // grad = 6.0 + ca2 * 0.105 + nkcc1 * 0.080
        __m256 v_grad_part = _mm256_fmadd_ps(v_ca, v_ca2_coeff, v_base_grad);
        __m256 v_grad = _mm256_fmadd_ps(v_nk, v_nkcc1_coeff, v_grad_part);
        _mm256_storeu_ps(out_grad + i, v_grad);

        // csf = (grad / 20.0) * 0.36
        __m256 v_csf = _mm256_mul_ps(v_grad, v_csf_scale);
        _mm256_storeu_ps(out_csf + i, v_csf);

        // icp = 7.0 + (csf / 0.35) * 4.5
        __m256 v_icp = _mm256_fmadd_ps(v_csf, v_icp_slope, v_icp_base);
        _mm256_storeu_ps(out_icp + i, v_icp);

        // comp = max(10.0, 100.0 - max(0.0, icp - 7.0) * 8.0)
        __m256 v_icp_excess = _mm256_max_ps(v_zero, _mm256_sub_ps(v_icp, v_icp_base));
        __m256 v_comp_raw = _mm256_sub_ps(v_hundred, _mm256_mul_ps(v_icp_excess, v_comp_slope));
        __m256 v_comp = _mm256_max_ps(v_min_comp, v_comp_raw);
        _mm256_storeu_ps(out_comp + i, v_comp);
    }
}

int main() {
    std::vector<float> ca2(N_CHANNELS, 85.0f);
    std::vector<float> nkcc1(N_CHANNELS, 80.0f);
    std::vector<float> grad(N_CHANNELS);
    std::vector<float> csf(N_CHANNELS);
    std::vector<float> icp(N_CHANNELS);
    std::vector<float> comp(N_CHANNELS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        benchmark_choroid_plexus_avx2(
            ca2.data(), nkcc1.data(),
            grad.data(), csf.data(), icp.data(), comp.data(),
            N_CHANNELS
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERATIONS; ++i) {
        benchmark_choroid_plexus_avx2(
            ca2.data(), nkcc1.data(),
            grad.data(), csf.data(), icp.data(), comp.data(),
            N_CHANNELS
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / N_ITERATIONS;
    double throughput_mch_s = (static_cast<double>(N_CHANNELS) / (avg_us * 1e-6)) / 1e6;

    std::cout << "[AVX2-FMA Benchmark] Choroid Plexus CSF Secretion Kinetics" << std::endl;
    std::cout << "Channels: " << N_CHANNELS << " | Iterations: " << N_ITERATIONS << std::endl;
    std::cout << "Avg latency per iteration: " << avg_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput_mch_s << " Million channels/sec" << std::endl;
    std::cout << "Sample verification [Channel 0]:" << std::endl;
    std::cout << "  CA-II Drive: " << ca2[0] << "% | NKCC1 Rate: " << nkcc1[0] << "%" << std::endl;
    std::cout << "  Osmotic Gradient: " << grad[0] << " mOsm | CSF Secretion: " << csf[0] << " mL/min" << std::endl;
    std::cout << "  Intracranial Pressure: " << icp[0] << " mmHg | Compliance Reserve: " << comp[0] << "%" << std::endl;

    return 0;
}
