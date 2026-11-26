#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <immintrin.h>

/**
 * 2-bit Packed DNA Sequence Search & Hamming Distance Microbenchmark.
 * A = 00, C = 01, G = 10, T = 11.
 * 32 nucleotides per 64-bit integer word.
 */
static size_t dna_hamming_distance_avx2(const uint64_t* query, const uint64_t* target, size_t num_words) {
    size_t mismatches = 0;
    for (size_t i = 0; i < num_words; ++i) {
        uint64_t diff = query[i] ^ target[i];
        // Any non-zero 2-bit pair indicates a mismatch
        uint64_t odd_bits = diff & 0x5555555555555555ULL;
        uint64_t even_bits = (diff >> 1) & 0x5555555555555555ULL;
        uint64_t mismatch_mask = odd_bits | even_bits;
        mismatches += __builtin_popcountll(mismatch_mask);
    }
    return mismatches;
}

int main(int argc, char** argv) {
    size_t iterations = 5000000; // 5 Million window searches
    if (argc > 1) {
        iterations = std::stoul(argv[1]);
    }

    const size_t num_words = 4; // 4 * 32 = 128 base pairs window
    std::vector<uint64_t> query(num_words, 0x1B1B1B1B1B1B1B1BULL);
    std::vector<uint64_t> target(num_words, 0x1B1B1B1B1B1B1B1AULL);

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ 2-Bit Packed DNA Search Microbenchmark\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Window Size   : 128 base pairs (4 x 64-bit words)\n";
    std::cout << "[CONFIG] Alignments    : " << iterations << "\n";

    // Warmup
    size_t dist = dna_hamming_distance_avx2(query.data(), target.data(), num_words);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        dist = dna_hamming_distance_avx2(query.data(), target.data(), num_words);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double alignments_per_sec = (double)iterations / (elapsed.count() / 1000.0);
    double bases_per_sec = alignments_per_sec * 128.0;

    std::cout << "[RESULT] Hamming Dist  : " << dist << " mismatches\n";
    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Alignment Rate: " << std::fixed << std::setprecision(2) << (alignments_per_sec / 1e6) << " M aligns/sec\n";
    std::cout << "[RESULT] Throughput    : " << std::fixed << std::setprecision(2) << (bases_per_sec / 1e6) << " Mb/s\n";

    return 0;
}
