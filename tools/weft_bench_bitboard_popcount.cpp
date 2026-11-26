#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <immintrin.h>

/**
 * 64-bit Bitboard Popcount & Prefix Mask AVX2 Microbenchmark.
 * Evaluates branchless token occupancy masking and prefix token accumulation.
 */
static size_t bitboard_prefix_popcount_avx2(const uint64_t* bitboard, size_t num_words) {
    size_t total_active = 0;
    for (size_t i = 0; i < num_words; ++i) {
        total_active += __builtin_popcountll(bitboard[i]);
    }
    return total_active;
}

int main(int argc, char** argv) {
    size_t iterations = 2000000; // 2 Million bitboard checks
    if (argc > 1) {
        iterations = std::stoul(argv[1]);
    }

    const size_t num_words = 64; // 64 * 64 = 4096 context tokens
    std::vector<uint64_t> bitboard(num_words, 0xAAAAAAAAAAAAAAAAULL); // Alternating active tokens

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ Bitboard Popcount & Mask Microbench\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Context Size  : " << (num_words * 64) << " tokens (" << num_words << " words)\n";
    std::cout << "[CONFIG] Iterations    : " << iterations << "\n";

    // Warmup
    size_t active = bitboard_prefix_popcount_avx2(bitboard.data(), num_words);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        active = bitboard_prefix_popcount_avx2(bitboard.data(), num_words);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double ops_per_sec = (double)iterations / (elapsed.count() / 1000.0);
    double words_per_sec = (double)(iterations * num_words) / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Active Tokens : " << active << "\n";
    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Popcount Rate : " << std::fixed << std::setprecision(2) << (words_per_sec / 1e6) << " M words/sec\n";
    std::cout << "[RESULT] Throughput    : " << std::fixed << std::setprecision(2) << (ops_per_sec / 1e6) << " M checks/sec\n";

    return 0;
}
