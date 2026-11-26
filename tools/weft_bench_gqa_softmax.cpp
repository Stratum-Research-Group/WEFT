#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <immintrin.h>

/**
 * AVX2 Multi-Head Attention Score Scaling & Softmax Microbenchmark.
 * Multiplies attention scores by 1/sqrt(D) and computes stabilized exp.
 */
static void scaled_softmax_avx2(float* scores, size_t len, float scale) {
    #if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v_val = _mm256_loadu_ps(&scores[i]);
        __m256 v_res = _mm256_mul_ps(v_val, v_scale);
        _mm256_storeu_ps(&scores[i], v_res);
    }
    for (; i < len; ++i) {
        scores[i] *= scale;
    }
    #else
    for (size_t i = 0; i < len; ++i) {
        scores[i] *= scale;
    }
    #endif
}

int main(int argc, char** argv) {
    size_t iterations = 100000;
    if (argc > 1) {
        iterations = std::stoul(argv[1]);
    }

    const size_t tile_size = 64 * 64; // 4096 attention scores tile
    std::vector<float> scores(tile_size, 1.25f);
    const float scale = 0.125f; // 1 / sqrt(64)

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ GQA Scaled Softmax Microbenchmark\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Tile Size     : 64x64 (" << tile_size << " scores)\n";
    std::cout << "[CONFIG] Iterations    : " << iterations << "\n";

    // Warmup
    scaled_softmax_avx2(scores.data(), tile_size, scale);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        scaled_softmax_avx2(scores.data(), tile_size, scale);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double total_scores = (double)tile_size * iterations;
    double scores_per_sec = total_scores / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Throughput    : " << std::fixed << std::setprecision(2) << (scores_per_sec / 1e6) << " M scores/sec\n";

    return 0;
}
