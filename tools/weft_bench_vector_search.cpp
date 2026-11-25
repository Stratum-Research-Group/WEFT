#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Compute cosine similarity between two float vectors of dimension D using AVX2
static float cosine_similarity_avx2(const float* a, const float* b, size_t dim) {
    __m256 v_dot = _mm256_setzero_ps();
    __m256 v_norm_a = _mm256_setzero_ps();
    __m256 v_norm_b = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);

        v_dot = _mm256_fmadd_ps(va, vb, v_dot);
        v_norm_a = _mm256_fmadd_ps(va, va, v_norm_a);
        v_norm_b = _mm256_fmadd_ps(vb, vb, v_norm_b);
    }

    alignas(32) float buf_dot[8], buf_na[8], buf_nb[8];
    _mm256_storeu_ps(buf_dot, v_dot);
    _mm256_storeu_ps(buf_na, v_norm_a);
    _mm256_storeu_ps(buf_nb, v_norm_b);

    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int k = 0; k < 8; ++k) {
        dot += buf_dot[k];
        na += buf_na[k];
        nb += buf_nb[k];
    }

    // Scalar tail
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0f ? (dot / denom) : 0.0f;
}

int main(int argc, char** argv) {
    size_t queries = 50000;
    if (argc > 1) {
        queries = std::stoul(argv[1]);
    }

    const size_t dim = 1024; // Standard embedding dimension
    std::vector<float> q(dim, 0.5f);
    std::vector<float> doc(dim, 0.4f);

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ SIMD Vector Cosine Search Microbenchmark\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Vector Dimension: " << dim << "\n";
    std::cout << "[CONFIG] Query Iterations: " << queries << "\n";

    // Warmup
    float res = cosine_similarity_avx2(q.data(), doc.data(), dim);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < queries; ++i) {
        res = cosine_similarity_avx2(q.data(), doc.data(), dim);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double qps = (double)queries / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Cosine Result   : " << std::fixed << std::setprecision(4) << res << "\n";
    std::cout << "[RESULT] Elapsed Time    : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Throughput      : " << std::fixed << std::setprecision(2) << qps << " queries/sec\n";

    return 0;
}
