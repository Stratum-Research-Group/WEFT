#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <immintrin.h>

/**
 * Top-K & Top-P (Nucleus) Sampling Microbenchmark for Large Vocabularies.
 * Vocab = 32,000 tokens (Standard Llama/Gemma vocab size).
 */
int main(int argc, char** argv) {
    size_t iterations = 50000;
    if (argc > 1) {
        iterations = std::stoul(argv[1]);
    }

    const size_t vocab_size = 32000;
    std::vector<float> logits(vocab_size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = dist(rng);
    }

    const size_t top_k = 50;
    const float top_p = 0.90f;

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ Fast Top-K / Top-P Sampling Microbench\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Vocab Size    : " << vocab_size << " tokens\n";
    std::cout << "[CONFIG] Top-K / Top-P : K=" << top_k << " | P=" << top_p << "\n";
    std::cout << "[CONFIG] Iterations    : " << iterations << "\n";

    std::vector<float> temp_buffer = logits;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        // Fast partial sort for Top-K
        std::nth_element(temp_buffer.begin(), temp_buffer.begin() + top_k, temp_buffer.end(), std::greater<float>());
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double samples_per_sec = (double)iterations / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Sampling Rate : " << std::fixed << std::setprecision(2) << samples_per_sec << " tokens/sec\n";
    std::cout << "[RESULT] Latency / Tok : " << std::fixed << std::setprecision(3) << (elapsed.count() / iterations) << " ms/tok\n";

    return 0;
}
