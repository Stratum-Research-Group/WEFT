#include "../csrc/kernels/avx512_subbit_engine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

int main() {
    std::cout << "=== AVX-512 Zen 4 Micro-Benchmark & Roofline Profiler ===" << std::endl;

    // Stream 64 MB of packed bitboards (representing 512 Million binary weights)
    const size_t num_words = 8 * 1024 * 1024; // 8M uint64_t = 64 MB
    std::vector<uint64_t> bitboards(num_words, 0xAAAAAAAAAAAAAAAAULL);
    std::vector<int8_t> activations(64, 1);

    const int iterations = 100;
    volatile int64_t total_acc = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it) {
        for (size_t w = 0; w < num_words; w += 64) {
            total_acc += weft::AVX512SubBitEngine::dot_product_binary(activations.data(), &bitboards[w], 64);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t processed_words = (num_words / 64) * iterations;
    size_t processed_weights = processed_words * 64;
    double bytes_streamed = processed_words * sizeof(uint64_t);
    double giga_ops = (processed_weights * 2.0) / (elapsed_sec * 1e9);
    double bandwidth_gb_s = (bytes_streamed / (elapsed_sec * 1e9));

    std::cout << "Processed: " << processed_weights << " binary weights in " << elapsed_sec << " s" << std::endl;
    std::cout << "Throughput: " << giga_ops << " GOPS" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

    std::ofstream out("results/avx512_zen4_roofline.json");
    out << "{\n"
        << "  \"cpu\": \"AMD Ryzen 5 7600X (Zen 4, AVX-512)\",\n"
        << "  \"iterations\": " << iterations << ",\n"
        << "  \"processed_weights\": " << processed_weights << ",\n"
        << "  \"elapsed_seconds\": " << elapsed_sec << ",\n"
        << "  \"giga_ops_per_second\": " << giga_ops << ",\n"
        << "  \"bandwidth_gb_per_sec\": " << bandwidth_gb_s << "\n"
        << "}\n";
    out.close();
    std::cout << "[PASS] Roofline results written to results/avx512_zen4_roofline.json" << std::endl;

    return 0;
}
