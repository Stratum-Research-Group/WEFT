// Prefrontal Cortex & Mediodorsal Thalamic Reciprocal Loop AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates persistent delay-period firing, reciprocal thalamocortical positive feedback, and bistability.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_PFC = 1024; // 1,024 Prefrontal pyramidal neurons
    const size_t N_MD  = 256;  // 256 Mediodorsal thalamic projection neurons
    const size_t DELAY_STEPS = 50; // 50 time steps across delay period
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT PREFRONTAL-MD THALAMIC WORKING MEMORY AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Dimensions: N_PFC=" << N_PFC << ", N_MD=" << N_MD << ", Delay Steps=" << DELAY_STEPS << std::endl;

    std::vector<float> W_pfc(N_PFC * N_PFC);
    std::vector<float> W_th_to_ctx(N_PFC * N_MD);
    std::vector<float> W_ctx_to_th(N_MD * N_PFC);
    std::vector<float> x_pfc(N_PFC);
    std::vector<float> x_md(N_MD);
    std::vector<float> x_pfc_next(N_PFC);
    std::vector<float> x_md_next(N_MD);

    std::mt19937 rng(42);
    std::normal_distribution<float> w_dist(0.0f, 1.0f / std::sqrt(N_PFC));
    std::uniform_real_distribution<float> act_dist(0.0f, 1.0f);

    for (size_t i = 0; i < N_PFC * N_PFC; ++i) W_pfc[i] = w_dist(rng);
    for (size_t i = 0; i < N_PFC * N_MD; ++i)  W_th_to_ctx[i] = w_dist(rng);
    for (size_t i = 0; i < N_MD * N_PFC; ++i)  W_ctx_to_th[i] = w_dist(rng);

    // Initial cue loading in PFC
    for (size_t i = 0; i < N_PFC; ++i) x_pfc[i] = act_dist(rng);
    for (size_t i = 0; i < N_MD; ++i)  x_md[i] = 0.0f;

    const float gamma_md = 2.5f;
    const float dt = 0.1f;
    const float inh = 0.15f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t s = 0; s < DELAY_STEPS; ++s) {
            // Update PFC
            for (size_t i = 0; i < N_PFC; ++i) {
                __m256 acc_pfc = _mm256_setzero_ps();
                for (size_t j = 0; j < N_PFC; j += 8) {
                    __m256 w = _mm256_loadu_ps(&W_pfc[i * N_PFC + j]);
                    __m256 x = _mm256_loadu_ps(&x_pfc[j]);
                    acc_pfc = _mm256_fmadd_ps(w, x, acc_pfc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc_pfc);
                float s_pfc = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

                __m256 acc_th = _mm256_setzero_ps();
                for (size_t j = 0; j < N_MD; j += 8) {
                    __m256 w = _mm256_loadu_ps(&W_th_to_ctx[i * N_MD + j]);
                    __m256 x = _mm256_loadu_ps(&x_md[j]);
                    acc_th = _mm256_fmadd_ps(w, x, acc_th);
                }
                alignas(32) float tmp_th[8];
                _mm256_storeu_ps(tmp_th, acc_th);
                float s_th = tmp_th[0] + tmp_th[1] + tmp_th[2] + tmp_th[3] + tmp_th[4] + tmp_th[5] + tmp_th[6] + tmp_th[7];

                float drive = s_pfc + gamma_md * s_th - inh;
                float act = drive > 0.0f ? drive : 0.0f;
                x_pfc_next[i] = x_pfc[i] + dt * (-x_pfc[i] + act);
            }
            // Update MD Thalamus
            for (size_t i = 0; i < N_MD; ++i) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t j = 0; j < N_PFC; j += 8) {
                    __m256 w = _mm256_loadu_ps(&W_ctx_to_th[i * N_PFC + j]);
                    __m256 x = _mm256_loadu_ps(&x_pfc[j]);
                    acc = _mm256_fmadd_ps(w, x, acc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                float s_ctx = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                float act = s_ctx > 0.0f ? s_ctx : 0.0f;
                x_md_next[i] = x_md[i] + dt * (-x_md[i] + act);
            }
            std::copy(x_pfc_next.begin(), x_pfc_next.end(), x_pfc.begin());
            std::copy(x_md_next.begin(), x_md_next.end(), x_md.begin());
        }
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t s = 0; s < DELAY_STEPS; ++s) {
            for (size_t i = 0; i < N_PFC; ++i) {
                __m256 acc_pfc = _mm256_setzero_ps();
                for (size_t j = 0; j < N_PFC; j += 8) {
                    __m256 w = _mm256_loadu_ps(&W_pfc[i * N_PFC + j]);
                    __m256 x = _mm256_loadu_ps(&x_pfc[j]);
                    acc_pfc = _mm256_fmadd_ps(w, x, acc_pfc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc_pfc);
                float s_pfc = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

                __m256 acc_th = _mm256_setzero_ps();
                for (size_t j = 0; j < N_MD; j += 8) {
                    __m256 w = _mm256_loadu_ps(&W_th_to_ctx[i * N_MD + j]);
                    __m256 x = _mm256_loadu_ps(&x_md[j]);
                    acc_th = _mm256_fmadd_ps(w, x, acc_th);
                }
                alignas(32) float tmp_th[8];
                _mm256_storeu_ps(tmp_th, acc_th);
                float s_th = tmp_th[0] + tmp_th[1] + tmp_th[2] + tmp_th[3] + tmp_th[4] + tmp_th[5] + tmp_th[6] + tmp_th[7];

                float drive = s_pfc + gamma_md * s_th - inh;
                float act = drive > 0.0f ? drive : 0.0f;
                x_pfc_next[i] = x_pfc[i] + dt * (-x_pfc[i] + act);
            }
            for (size_t i = 0; i < N_MD; ++i) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t j = 0; j < N_PFC; j += 8) {
                    __m256 w = _mm256_loadu_ps(&W_ctx_to_th[i * N_PFC + j]);
                    __m256 x = _mm256_loadu_ps(&x_pfc[j]);
                    acc = _mm256_fmadd_ps(w, x, acc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                float s_ctx = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                float act = s_ctx > 0.0f ? s_ctx : 0.0f;
                x_md_next[i] = x_md[i] + dt * (-x_md[i] + act);
            }
            std::copy(x_pfc_next.begin(), x_pfc_next.end(), x_pfc.begin());
            std::copy(x_md_next.begin(), x_md_next.end(), x_md.begin());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t fma_ops = ITERS * DELAY_STEPS * (N_PFC * N_PFC * 2 + N_PFC * N_MD * 2 + N_MD * N_PFC * 2);
    double tflops = (double)fma_ops / elapsed_sec / 1e12;
    double step_latency_us = (elapsed_sec / (ITERS * DELAY_STEPS)) * 1e6;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "PFC-MD Working Memory FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "Reciprocal Loop Step Latency: " << step_latency_us << " us/step" << std::endl;

    return 0;
}
