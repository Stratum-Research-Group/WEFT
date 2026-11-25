#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <immintrin.h>

// Standalone C++ AVX-512 / AVX2 Paged Flash-Decoding Microbenchmark
// Simulates multi-query attention head verification over 16-token memory pages

constexpr size_t NUM_PAGES = 1024;
constexpr size_t TOKENS_PER_PAGE = 16;
constexpr size_t HEAD_DIM = 64; // D = 64

struct alignas(64) PagedCache {
    float k_pages[NUM_PAGES][TOKENS_PER_PAGE][HEAD_DIM];
    float v_pages[NUM_PAGES][TOKENS_PER_PAGE][HEAD_DIM];
};

void run_paged_flash_decoding(const float* q, const PagedCache& cache, float* out, size_t active_pages) {
    float max_score = -1e9f;
    float sum_exp = 0.0f;
    std::vector<float> acc(HEAD_DIM, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    for (size_t p = 0; p < active_pages; ++p) {
        for (size_t t = 0; t < TOKENS_PER_PAGE; ++t) {
            float dot = 0.0f;
#if defined(__AVX512F__)
            __m512 vdot = _mm512_setzero_ps();
            for (size_t d = 0; d < HEAD_DIM; d += 16) {
                __m512 vq = _mm512_loadu_ps(q + d);
                __m512 vk = _mm512_loadu_ps(&cache.k_pages[p][t][d]);
                vdot = _mm512_fmadd_ps(vq, vk, vdot);
            }
            dot = _mm512_reduce_add_ps(vdot);
#elif defined(__AVX2__)
            __m256 vdot1 = _mm256_setzero_ps();
            __m256 vdot2 = _mm256_setzero_ps();
            for (size_t d = 0; d < HEAD_DIM; d += 16) {
                __m256 vq1 = _mm256_loadu_ps(q + d);
                __m256 vk1 = _mm256_loadu_ps(&cache.k_pages[p][t][d]);
                vdot1 = _mm256_fmadd_ps(vq1, vk1, vdot1);
                __m256 vq2 = _mm256_loadu_ps(q + d + 8);
                __m256 vk2 = _mm256_loadu_ps(&cache.k_pages[p][t][d + 8]);
                vdot2 = _mm256_fmadd_ps(vq2, vk2, vdot2);
            }
            __m256 vdot = _mm256_add_ps(vdot1, vdot2);
            alignas(32) float buf[8];
            _mm256_storeu_ps(buf, vdot);
            for (int i = 0; i < 8; ++i) dot += buf[i];
#else
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                dot += q[d] * cache.k_pages[p][t][d];
            }
#endif
            dot *= scale;
            if (dot > max_score) {
                float shift = std::exp(max_score - dot);
                sum_exp = sum_exp * shift + 1.0f;
                for (size_t d = 0; d < HEAD_DIM; ++d) {
                    acc[d] = acc[d] * shift + cache.v_pages[p][t][d];
                }
                max_score = dot;
            } else {
                float p_val = std::exp(dot - max_score);
                sum_exp += p_val;
                for (size_t d = 0; d < HEAD_DIM; ++d) {
                    acc[d] += p_val * cache.v_pages[p][t][d];
                }
            }
        }
    }

    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    for (size_t d = 0; d < HEAD_DIM; ++d) {
        out[d] = acc[d] * inv_sum;
    }
}

int main() {
    std::cout << "=== WEFT SIMD Paged Flash-Decoding Microbenchmark ===" << std::endl;
    auto cache = std::make_unique<PagedCache>();
    std::vector<float> q(HEAD_DIM, 0.1f);
    std::vector<float> out(HEAD_DIM, 0.0f);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t p = 0; p < NUM_PAGES; ++p) {
        for (size_t t = 0; t < TOKENS_PER_PAGE; ++t) {
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                cache->k_pages[p][t][d] = dist(rng);
                cache->v_pages[p][t][d] = dist(rng);
            }
        }
    }

    size_t active_pages = 256; // 4,096 tokens in context
    constexpr size_t ITERS = 1000;

    // Warmup
    for (size_t i = 0; i < 50; ++i) {
        run_paged_flash_decoding(q.data(), *cache, out.data(), active_pages);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERS; ++i) {
        run_paged_flash_decoding(q.data(), *cache, out.data(), active_pages);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double us_per_token = (elapsed_ms * 1000.0) / ITERS;
    double tokens_per_sec = (1000.0 * 1000.0) / us_per_token;

    std::cout << "[RESULT] Context Tokens: " << active_pages * TOKENS_PER_PAGE << std::endl;
    std::cout << "[RESULT] Total Iterations: " << ITERS << std::endl;
    std::cout << "[RESULT] Latency per verification step: " << us_per_token << " us" << std::endl;
    std::cout << "[RESULT] Verification Throughput: " << tokens_per_sec << " steps/sec" << std::endl;
    std::cout << "[RESULT] Verification Checksum: " << out[0] << ", " << out[1] << std::endl;

    return 0;
}
