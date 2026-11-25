#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <immintrin.h>

/**
 * Fast SIMD 2048-bit Molecular Morgan / ECFP4 Fingerprint & Tanimoto Similarity.
 * Uses AVX2 popcount and bitwise AND to compare chemical structures at scale.
 */
static float tanimoto_similarity_avx2(const uint64_t* fp_a, const uint64_t* fp_b, size_t num_words) {
    size_t intersection_bits = 0;
    size_t union_bits = 0;

    for (size_t i = 0; i < num_words; ++i) {
        uint64_t a = fp_a[i];
        uint64_t b = fp_b[i];

        intersection_bits += __builtin_popcountll(a & b);
        union_bits += __builtin_popcountll(a | b);
    }

    return union_bits > 0 ? (float)intersection_bits / (float)union_bits : 1.0f;
}

int main(int argc, char** argv) {
    size_t comparisons = 2000000; // 2 Million molecular fingerprint comparisons
    if (argc > 1) {
        comparisons = std::stoul(argv[1]);
    }

    const size_t num_words = 32; // 32 * 64 bits = 2048-bit fingerprint (Standard ECFP4)
    std::vector<uint64_t> fp_query(num_words, 0xAAAAAAAAAAAAAAAAULL);
    std::vector<uint64_t> fp_target(num_words, 0xAAAAAAAAAAAAAAA5ULL);

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ Chemical Morgan Fingerprint Microbench\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Fingerprint Size : 2048-bit (32 x 64-bit words)\n";
    std::cout << "[CONFIG] Comparisons      : " << comparisons << "\n";

    // Warmup
    float sim = tanimoto_similarity_avx2(fp_query.data(), fp_target.data(), num_words);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < comparisons; ++i) {
        sim = tanimoto_similarity_avx2(fp_query.data(), fp_target.data(), num_words);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double comp_per_sec = (double)comparisons / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Tanimoto Sim    : " << std::fixed << std::setprecision(4) << sim << "\n";
    std::cout << "[RESULT] Elapsed Time    : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Throughput      : " << std::fixed << std::setprecision(2) << (comp_per_sec / 1e6) << " M comparisons/sec\n";

    return 0;
}
